"""Spotify playback, through the Web API via spotipy.

Deliberately *not* an MCP server, for the same reason as Calendar: a live OAuth
token for a real account stays inside code you own.

**Playback only** — play, queue, pause, resume, skip. No volume tool, no liking
tracks or editing playlists. Playing the wrong song is undone in a second;
a misheard "add this to my running playlist" leaves a quiet mess nobody
notices for weeks. That's the same blast-radius argument that keeps Calendar
without a delete tool.

**No confirmation step**, unlike Calendar's writes. Every action here is
instantly reversible, and a desk robot that asks "are you sure?" before
skipping a track is a desk robot you stop talking to.

The Web API doesn't play audio; it tells a Spotify device what to play, and
it requires Premium. When the bot's own player is set up (see
`brain.spotify_player`), play and resume always go to it — no guessing.
Otherwise the target is the active device, else the one named by
`SPOTIFY_DEVICE_NAME` (an optional tiebreaker), else whichever open device
Spotify lists first.

Credentials live in `.env`, encrypted like Calendar's (see `brain.config`).
Unlike Calendar, the token can't just be refreshed once at startup: Spotify
access tokens last an hour and the Brain runs for days. So spotipy refreshes
on demand through a cache handler whose "cache" *is* `SPOTIFY_TOKEN_JSON`,
re-encrypted and rewritten in place via `set_env_value`.

Known limit: Spotify blocks new API apps from its own algorithmic playlists
(Discover Weekly and friends). Neil's own playlists work; those may not.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from brain.config import set_env_value

__all__ = [
    "REDIRECT_URI",
    "SCOPES",
    "EnvTokenStore",
    "MusicDucker",
    "build_spotify_dispatch",
    "connect_spotify",
    "make_auth_manager",
    "spotify_schemas",
]

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]

SCOPES = ["user-modify-playback-state", "user-read-playback-state", "playlist-read-private"]

# Spotify no longer accepts `localhost` redirect URIs — it must be the
# loopback IP literal, and must match the app's dashboard setting exactly.
REDIRECT_URI = "http://127.0.0.1:8888/callback"

KINDS = ("track", "artist", "album", "playlist")

# How long to wait for the bot's own player to show up in Spotify's device
# list after (re)starting it. librespot usually registers in 2-4 seconds.
PLAYER_WAIT_SECONDS = 15.0
PLAYER_POLL_SECONDS = 0.5

# Music level while DmillsGPT is talking, in Spotify's 0-100 volume.
DUCK_VOLUME_PERCENT = 20


class EnvTokenStore:
    """The Spotify token, held in memory and persisted, encrypted, to `.env`.

    Plain class so it's testable without spotipy installed; `make_auth_manager`
    adapts it to spotipy's `CacheHandler` interface.
    """

    def __init__(self, env_file: Path, token_json: str | None) -> None:
        self.env_file = env_file
        self._token: dict[str, Any] | None = json.loads(token_json) if token_json else None

    def get(self) -> dict[str, Any] | None:
        return self._token

    def save(self, token: dict[str, Any]) -> None:
        # Persist the whole thing, not just the access token: Spotify sometimes
        # rotates the refresh token too, and losing that means re-authorizing.
        self._token = token
        set_env_value(self.env_file, "SPOTIFY_TOKEN_JSON", json.dumps(token))


def make_auth_manager(
    client_id: str, client_secret: str, store: EnvTokenStore, *, open_browser: bool
) -> Any:
    """A spotipy `SpotifyOAuth` whose token cache is `store`."""
    from spotipy.cache_handler import CacheHandler
    from spotipy.oauth2 import SpotifyOAuth

    class _EnvCacheHandler(CacheHandler):  # type: ignore[misc]
        def get_cached_token(self) -> dict[str, Any] | None:
            return store.get()

        def save_token_to_cache(self, token_info: dict[str, Any]) -> None:
            store.save(token_info)

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_handler=_EnvCacheHandler(),
        open_browser=open_browser,
    )


def connect_spotify(
    client_id: str | None,
    client_secret: str | None,
    token_json: str | None,
    env_file: Path,
) -> Any | None:
    """Build an authorized Spotify client, or None when not set up yet.

    Mirrors `connect_calendar`: no client id/secret means no Spotify and the
    bot runs as before; id/secret without a token is a real setup gap and
    raises with instructions.
    """
    if not client_id or not client_secret:
        return None

    if not token_json:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID/SECRET are set but there's no token yet. Run: "
            "python -m brain.authorize_spotify (opens a browser once, writes "
            "SPOTIFY_TOKEN_JSON to your .env)."
        )

    import spotipy

    store = EnvTokenStore(env_file, token_json)
    auth_manager = make_auth_manager(client_id, client_secret, store, open_browser=False)

    # Checked once, here, on purpose: if spotipy ever finds no usable token
    # mid-conversation it falls back to an interactive input() prompt, which
    # would hang the Brain waiting on a terminal nobody is looking at.
    if auth_manager.validate_token(store.get()) is None:
        raise RuntimeError(
            "The Spotify token is invalid and can't be refreshed. Run: "
            "python -m brain.authorize_spotify to re-authorize."
        )

    return spotipy.Spotify(auth_manager=auth_manager)


def spotify_schemas() -> list[dict[str, Any]]:
    """OpenAI `tools` entries for the Spotify tools."""
    no_args: dict[str, Any] = {"type": "object", "properties": {}}
    return [
        {
            "type": "function",
            "function": {
                "name": "play_music",
                "description": (
                    "Play something on Neil's Spotify right now. Just do it — no need "
                    "to confirm first. Use kind=artist for 'play some Kendrick', "
                    "kind=album for an album, kind=playlist for a named playlist, and "
                    "kind=track (the default) for a specific song."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for, e.g. 'GNX Kendrick Lamar'.",
                        },
                        "kind": {
                            "type": "string",
                            "enum": list(KINDS),
                            "description": "What the query names. Defaults to track.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pause_music",
                "description": (
                    "Pause Spotify. Use this for 'stop' and 'turn it off' too — "
                    "Spotify has no separate stop."
                ),
                "parameters": no_args,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resume_music",
                "description": "Resume Spotify from where it was paused.",
                "parameters": no_args,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "skip_track",
                "description": "Skip to the next track on Spotify.",
                "parameters": no_args,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "queue_track",
                "description": (
                    "Add one or more songs to play after the current one, without "
                    "interrupting it. Use for 'queue X', 'play X next', 'add X, Y and "
                    "Z after this'. Put EVERY requested song in one call, in the order "
                    "asked — never one call per song. Songs only: Spotify's queue "
                    "can't take a whole artist or album. If nothing is playing, use "
                    "play_music for the first song instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "songs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Each song as a search query, in play order, e.g. "
                                "['Luther Kendrick Lamar', 'TV Off']."
                            ),
                        }
                    },
                    "required": ["songs"],
                },
            },
        },
    ]


class MusicDucker:
    """Turns Spotify down while the bot talks, then puts it back.

    Voice mode only — the text CLI never speaks, so never ducks. Works on
    whichever device is playing, not just the bot's own player: music on
    your phone talking over the robot is the same problem.
    """

    def __init__(self, client: Any, level: int = DUCK_VOLUME_PERCENT) -> None:
        self._client = client
        self._level = level

    def duck(self) -> tuple[str | None, int] | None:
        """Lower the volume if music is playing. Returns what to restore."""
        playback = self._client.current_playback()
        if not playback or not playback.get("is_playing"):
            return None
        device = playback.get("device") or {}
        volume = device.get("volume_percent")
        if volume is None or not device.get("supports_volume", True) or volume <= self._level:
            return None
        device_id = device.get("id")
        self._client.volume(self._level, device_id=device_id)
        return device_id, int(volume)

    def restore(self, saved: tuple[str | None, int]) -> None:
        device_id, volume = saved
        self._client.volume(volume, device_id=device_id)

    @asynccontextmanager
    async def ducked(self) -> AsyncIterator[None]:
        # Ducking is a nicety; it must never cost a spoken reply. Any Spotify
        # failure here is swallowed and the bot just talks over the music.
        try:
            saved = await asyncio.to_thread(self.duck)
        except Exception:
            saved = None
        try:
            yield
        finally:
            if saved is not None:
                with suppress(Exception):
                    await asyncio.to_thread(self.restore, saved)


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c' — how a list sounds read aloud."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _describe_error(exc: Exception) -> str:
    """A Spotify failure as something the model can say out loud."""
    status = getattr(exc, "http_status", None)
    if status == 403:
        return "Spotify refused — playback control needs a Premium account."
    if status == 404:
        return "Nothing's playing on any device right now."
    msg = getattr(exc, "msg", None) or str(exc)
    return f"Spotify said: {msg}"


def build_spotify_dispatch(
    client: Any,
    default_device: str | None,
    player: Any = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    wait_seconds: float = PLAYER_WAIT_SECONDS,
) -> dict[str, ToolFn]:
    """Map the Spotify tool names to implementations, closing over `client`.

    `player`, when given, is the bot's own device (anything with `.name` and
    `.ensure_running()` — see `brain.spotify_player.LibrespotPlayer`).
    """

    def _own_player_device() -> dict[str, Any] | str:
        """The bot's player as a Spotify device, (re)starting it if needed."""
        player.ensure_running()
        deadline = wait_seconds
        while True:
            for device in (client.devices() or {}).get("devices") or []:
                if device.get("name") == player.name:
                    return dict(device)
            if deadline <= 0:
                return (
                    f"My Spotify player ({player.name}) didn't come up. "
                    f"Check {getattr(player, 'log_path', 'the librespot log')}."
                )
            sleep(PLAYER_POLL_SECONDS)
            deadline -= PLAYER_POLL_SECONDS

    def _target_device() -> dict[str, Any] | str:
        """The device to play on, or a sentence saying why there isn't one.

        Active device first, then `default_device` if set, then simply the
        first open one Spotify lists. `default_device` is only a tiebreaker
        for when several are open — never a requirement.
        """
        if player is not None:
            return _own_player_device()
        devices = [
            d for d in (client.devices() or {}).get("devices") or []
            # Restricted devices (some speakers, TVs) refuse Web API control.
            if not d.get("is_restricted")
        ]
        if not devices:
            return "Nothing's running Spotify. Open it on your Mac, phone or browser first."
        for device in devices:
            if device.get("is_active"):
                return dict(device)
        if default_device:
            wanted = default_device.strip().lower()
            for device in devices:
                if wanted in str(device.get("name", "")).lower():
                    return dict(device)
        # Every reply names the device it used, so a wrong pick is heard
        # immediately rather than playing silently somewhere else.
        return dict(devices[0])

    def _own_playlist(query: str) -> dict[str, Any] | None:
        wanted = query.strip().lower()
        items = (client.current_user_playlists(limit=50) or {}).get("items") or []
        for playlist in items:
            if playlist and str(playlist.get("name", "")).strip().lower() == wanted:
                return dict(playlist)
        return None

    def _search(query: str, kind: str) -> dict[str, Any] | None:
        result = client.search(q=query, limit=1, type=kind) or {}
        # Spotify sometimes returns null entries in playlist results.
        items = [i for i in (result.get(f"{kind}s") or {}).get("items") or [] if i]
        return dict(items[0]) if items else None

    def _label(item: dict[str, Any]) -> str:
        name = str(item.get("name") or "that")
        artists = item.get("artists") or []
        if artists:
            return f"{name} by {artists[0].get('name', 'someone')}"
        owner = (item.get("owner") or {}).get("display_name")
        return f"{name} ({owner}'s playlist)" if owner else name

    def _play(query: str, kind: str) -> str:
        device = _target_device()
        if isinstance(device, str):
            return device

        item = _own_playlist(query) if kind == "playlist" else None
        item = item or _search(query, kind)
        if item is None:
            return f"Couldn't find a {kind} matching {query!r} on Spotify."

        if kind == "track":
            client.start_playback(device_id=device["id"], uris=[item["uri"]])
        else:
            client.start_playback(device_id=device["id"], context_uri=item["uri"])
        return f"Playing {_label(item)} on {device.get('name', 'your device')}."

    def _resume() -> str:
        device = _target_device()
        if isinstance(device, str):
            return device
        client.start_playback(device_id=device["id"])
        return f"Resumed on {device.get('name', 'your device')}."

    def _pause() -> str:
        client.pause_playback()
        return "Paused."

    def _skip() -> str:
        client.next_track()
        return "Skipped."

    def _queue(songs: list[str]) -> str:
        queued: list[str] = []
        missing: list[str] = []
        for query in songs:
            item = _search(query, "track")
            if item is None:
                missing.append(query)
                continue
            try:
                # No device id: the queue belongs to whatever is playing now.
                client.add_to_queue(item["uri"])
            except Exception as exc:
                # Usually "nothing playing", which fails every song alike —
                # stop, and say what did make it in before the failure.
                done = f"Queued {_join(queued)}, then: " if queued else ""
                return done + _describe_error(exc)
            queued.append(_label(item))

        parts = []
        if queued:
            parts.append(f"Queued {_join(queued)}.")
        if missing:
            parts.append(f"Couldn't find {_join([repr(m) for m in missing])} on Spotify.")
        return " ".join(parts)

    async def _run(fn: Callable[..., str], *args: Any) -> str:
        # spotipy is synchronous; a blocking HTTP call inside the agent loop
        # would stall the voice path mid-turn. And a failed call shouldn't
        # kill the turn — the model can say what went wrong instead.
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:
            return _describe_error(exc)

    async def play_music(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "No song or artist given."
        kind = str(args.get("kind") or "track").strip().lower()
        if kind not in KINDS:
            return f"Unknown kind {kind!r}. Choose from: {', '.join(KINDS)}."
        return await _run(_play, query, kind)

    async def pause_music(args: dict[str, Any]) -> str:
        return await _run(_pause)

    async def resume_music(args: dict[str, Any]) -> str:
        return await _run(_resume)

    async def skip_track(args: dict[str, Any]) -> str:
        return await _run(_skip)

    async def queue_track(args: dict[str, Any]) -> str:
        raw = args.get("songs")
        if raw is None and args.get("query"):
            raw = [args["query"]]  # a model that ignored the schema's list
        if isinstance(raw, str):
            raw = [raw]
        songs = [str(s).strip() for s in raw or [] if str(s).strip()]
        if not songs:
            return "No song given."
        return await _run(_queue, songs)

    return {
        "play_music": play_music,
        "pause_music": pause_music,
        "resume_music": resume_music,
        "skip_track": skip_track,
        "queue_track": queue_track,
    }
