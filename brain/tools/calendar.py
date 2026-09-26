"""Google Calendar, through Google's own client libraries.

Deliberately *not* an MCP server. A third-party calendar MCP server runs in
your process holding live OAuth tokens for your real calendar — the trust
boundary is someone else's release cadence. This keeps that boundary inside
code you own. The generic MCP client comes later, for servers where third-party
code is the actual point.

Credentials are three encrypted values (`GOOGLE_CLIENT_ID`,
`GOOGLE_CLIENT_SECRET`, `GOOGLE_TOKEN_JSON`) in `.env`, not a client-secrets
JSON file or a separate token file — one place to look, one file to protect;
see `brain.config` for the encryption. The token is the one value here that
legitimately changes at runtime: it's re-encrypted and rewritten in place via
`set_env_value` after each refresh.

**Scope is `calendar.events`** — the narrowest scope Google offers that still
permits writes. It cannot read Drive, read Gmail, change calendar sharing, or
enumerate your other calendars. Everything here operates on `primary`.

**Still no delete tool.** This is driven by a voice loop that mishears things.
Creating a wrong event leaves a visible artifact you can spot and remove;
silently destroying a real one does not.

**Update exists, scoped to title and time only** — not location, not color,
not attendees. A misheard update to a real existing event is a real risk, but
narrower fields mean a narrower blast radius, and "move this to 5" is common
enough to be worth that risk. Deleting stays out for now.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.config import set_env_value

__all__ = [
    "SCOPES",
    "build_calendar_dispatch",
    "calendar_schemas",
    "connect_calendar",
    "format_events",
    "summarize_event",
]

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# A spoken reply cannot carry a month of appointments. Past this, the model is
# reading a list nobody can follow.
MAX_EVENTS = 20
DEFAULT_DAYS_AHEAD = 7

# Google's fixed event color palette (name -> colorId). This isn't
# discoverable per-account — it's the same eleven colors for every Google
# Calendar, documented at developers.google.com/calendar/api/v3/reference/colors.
EVENT_COLORS: dict[str, str] = {
    "lavender": "1",
    "sage": "2",
    "grape": "3",
    "flamingo": "4",
    "banana": "5",
    "tangerine": "6",
    "peacock": "7",
    "graphite": "8",
    "blueberry": "9",
    "basil": "10",
    "tomato": "11",
}


def _parse_local(text: str) -> datetime:
    """Parse an ISO 8601 string, attaching the system's local timezone if the
    model didn't include one.

    The model has no idea what timezone Neil is in, so it sends plain
    "2026-09-21T15:00:00" for "3pm" — a naive datetime. Google's API rejects
    that outright ("Missing time zone definition"), and even if it didn't,
    sending it bare would leave Google guessing which timezone 3pm meant.
    `.astimezone()` on a naive datetime attaches the *system's* local zone
    without changing the wall-clock time — exactly "3pm here," which is what
    was actually meant.
    """
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.astimezone()


def summarize_event(event: dict[str, Any]) -> str:
    """One event as a single readable line, including its id.

    All-day events carry `date`; timed events carry `dateTime`. Google returns
    one or the other, never both, and conflating them produces "your dentist
    appointment is at midnight".

    The id is never meant to be spoken aloud — it's how a later
    `update_calendar_event` call knows which event to change. A model that
    wants to move "the dentist thing" first has to have seen this list.
    """
    start = event.get("start", {})
    when = start.get("dateTime") or start.get("date") or "unknown time"
    title = event.get("summary") or "(no title)"
    location = event.get("location")
    event_id = event.get("id")
    prefix = f"[{event_id}] " if event_id else ""
    return f"{prefix}{when} — {title}" + (f" @ {location}" if location else "")


def format_events(events: Sequence[dict[str, Any]]) -> str:
    """Render the model's view of a calendar query."""
    if not events:
        return "Nothing on the calendar for that window."
    return "\n".join(f"- {summarize_event(event)}" for event in events)


def connect_calendar(
    client_id: str | None,
    client_secret: str | None,
    token_json: str | None,
    env_file: Path,
) -> Any | None:
    """Build an authorized Calendar service, or None when not set up yet.

    Returns None rather than raising when the client id/secret aren't in
    .env: a fresh checkout has no Google credentials and the bot must still
    run. Once those exist but there's no token yet, that's a real setup gap —
    it raises with instructions rather than silently running without Calendar.
    """
    if not client_id or not client_secret:
        return None

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not token_json:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID/SECRET are set but there's no token yet. Run: "
            "python -m brain.authorize_google (opens a browser once, writes "
            "GOOGLE_TOKEN_JSON to your .env)."
        )

    creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        set_env_value(env_file, "GOOGLE_TOKEN_JSON", creds.to_json())

    if not creds.valid:
        raise RuntimeError(
            "The Google Calendar token is invalid and can't be refreshed. Run: "
            "python -m brain.authorize_google to re-authorize."
        )

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def calendar_schemas() -> list[dict[str, Any]]:
    """OpenAI `tools` entries for the calendar tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_calendar_events",
                "description": (
                    "Look at Neil's actual calendar. Use this for anything about his "
                    "schedule, availability, or what's coming up — never guess at it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days_ahead": {
                            "type": "integer",
                            "description": (
                                f"How many days forward to look. Defaults to "
                                f"{DEFAULT_DAYS_AHEAD}. Use 1 for 'today'."
                            ),
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": (
                    "Put a new event on Neil's real calendar. This is a real write he "
                    "will see, so confirm the details with him before calling it — "
                    "especially if you inferred the time rather than being told it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start": {
                            "type": "string",
                            "description": "ISO 8601 start, e.g. 2026-09-21T15:00:00",
                        },
                        "end": {
                            "type": "string",
                            "description": (
                                "ISO 8601 end. Omit for a 1-hour event starting at `start`."
                            ),
                        },
                        "location": {"type": "string"},
                        "color": {
                            "type": "string",
                            "enum": sorted(EVENT_COLORS),
                            "description": "Optional event color tab. Only set this if asked.",
                        },
                    },
                    "required": ["title", "start"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_calendar_event",
                "description": (
                    "Change the title and/or time of an event that already exists on "
                    "Neil's calendar. You need its id from get_calendar_events — call "
                    "that first if you don't already have it from this conversation. "
                    "A real write to a real event, so confirm with him before calling "
                    "this, same as creating one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "The event's id, from get_calendar_events.",
                        },
                        "title": {
                            "type": "string",
                            "description": "New title. Omit to leave it unchanged.",
                        },
                        "start": {
                            "type": "string",
                            "description": (
                                "New ISO 8601 start. Omit to leave it unchanged. If you "
                                "give a new start but no new end, the event's original "
                                "length is preserved rather than just moving the start."
                            ),
                        },
                        "end": {
                            "type": "string",
                            "description": "New ISO 8601 end. Omit to leave it unchanged.",
                        },
                    },
                    "required": ["event_id"],
                },
            },
        },
    ]


def build_calendar_dispatch(service: Any) -> dict[str, ToolFn]:
    """Map the calendar tool names to implementations, closing over `service`."""

    def _list(days_ahead: int) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        response = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=(now + timedelta(days=days_ahead)).isoformat(),
                maxResults=MAX_EVENTS,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return list(response.get("items", []))

    async def get_calendar_events(args: dict[str, Any]) -> str:
        days = int(args.get("days_ahead") or DEFAULT_DAYS_AHEAD)
        # googleapiclient is synchronous; a blocking HTTP call inside the agent
        # loop would stall the voice path mid-turn.
        events = await asyncio.to_thread(_list, days)
        return format_events(events)

    def _insert(body: dict[str, Any]) -> dict[str, Any]:
        result = service.events().insert(calendarId="primary", body=body).execute()
        return dict(result)

    async def create_calendar_event(args: dict[str, Any]) -> str:
        title = str(args["title"]).strip()
        start = str(args["start"]).strip()
        end = str(args.get("end") or "").strip()

        try:
            start_dt = _parse_local(start)
        except ValueError:
            return f"Couldn't read {start!r} as a date and time. Use ISO 8601."

        if end:
            try:
                end_dt = _parse_local(end)
            except ValueError:
                return f"Couldn't read {end!r} as a date and time. Use ISO 8601."
        else:
            end_dt = start_dt + timedelta(hours=1)

        if end_dt <= start_dt:
            return "That event would end before it starts. Check the times."

        body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat()},
            "end": {"dateTime": end_dt.isoformat()},
        }
        if args.get("location"):
            body["location"] = str(args["location"]).strip()
        if args.get("color"):
            color = str(args["color"]).strip().lower()
            color_id = EVENT_COLORS.get(color)
            if color_id is None:
                return f"Unknown color {color!r}. Choose from: {', '.join(sorted(EVENT_COLORS))}."
            body["colorId"] = color_id

        created = await asyncio.to_thread(_insert, body)
        return f"Added: {summarize_event(created)}"

    def _get(event_id: str) -> dict[str, Any]:
        return dict(service.events().get(calendarId="primary", eventId=event_id).execute())

    def _patch(event_id: str, body: dict[str, Any]) -> dict[str, Any]:
        result = (
            service.events()
            .patch(calendarId="primary", eventId=event_id, body=body)
            .execute()
        )
        return dict(result)

    async def update_calendar_event(args: dict[str, Any]) -> str:
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            return "No event id given — call get_calendar_events first to find it."

        new_start = str(args.get("start") or "").strip()
        new_end = str(args.get("end") or "").strip()

        start_dt: datetime | None = None
        if new_start:
            try:
                start_dt = _parse_local(new_start)
            except ValueError:
                return f"Couldn't read {new_start!r} as a date and time. Use ISO 8601."

        end_dt: datetime | None = None
        if new_end:
            try:
                end_dt = _parse_local(new_end)
            except ValueError:
                return f"Couldn't read {new_end!r} as a date and time. Use ISO 8601."

        # A new start with no new end must not silently leave the end at its
        # old absolute time — "push it to 5" on a 3-4pm meeting would produce
        # a 5-4 event. Preserve the original duration instead.
        if start_dt is not None and end_dt is None:
            try:
                existing = await asyncio.to_thread(_get, event_id)
            except Exception as exc:
                return f"Couldn't find that event: {exc}"
            old_start = (existing.get("start") or {}).get("dateTime")
            old_end = (existing.get("end") or {}).get("dateTime")
            if old_start and old_end:
                duration = datetime.fromisoformat(old_end) - datetime.fromisoformat(old_start)
                end_dt = start_dt + duration

        if start_dt is not None and end_dt is not None and end_dt <= start_dt:
            return "That would end before it starts. Check the times."

        body: dict[str, Any] = {}
        if args.get("title"):
            body["summary"] = str(args["title"]).strip()
        if start_dt is not None:
            body["start"] = {"dateTime": start_dt.isoformat()}
        if end_dt is not None:
            body["end"] = {"dateTime": end_dt.isoformat()}

        if not body:
            return "Nothing to change — give a new title or time."

        try:
            updated = await asyncio.to_thread(_patch, event_id, body)
        except Exception as exc:
            return f"Couldn't update that event: {exc}"
        return f"Updated: {summarize_event(updated)}"

    return {
        "get_calendar_events": get_calendar_events,
        "create_calendar_event": create_calendar_event,
        "update_calendar_event": update_calendar_event,
    }
