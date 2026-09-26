"""One-time Google Calendar authorization.

Run once, interactively. It opens a browser, you consent, and it writes
`GOOGLE_TOKEN_JSON` straight into your `.env`, encrypted — no separate token
file, no client-secrets JSON on disk. The bot re-reads it (and re-encrypts it
in place after each refresh) from there onward.

    python -m brain.authorize_google

Kept out of the agent path on purpose: a bot that can pop a browser consent
screen mid-conversation is a bot that hangs waiting for a click nobody saw.
"""

from __future__ import annotations

import sys

from brain.config import Settings, set_env_value
from brain.tools.calendar import SCOPES


def main() -> int:
    settings = Settings.load()

    if not settings.google_client_id or not settings.google_client_secret:
        print(
            f"GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET aren't set in {settings.env_file}.\n\n"
            f"In Google Cloud Console: create a project, enable the Google Calendar API,\n"
            f"then create an OAuth client ID of type 'Desktop app'. Open the downloaded\n"
            f"JSON and add its client_id and client_secret to your .env, encrypted:\n\n"
            f"  npx @dotenvx/dotenvx set GOOGLE_CLIENT_ID\n"
            f"  npx @dotenvx/dotenvx set GOOGLE_CLIENT_SECRET\n\n"
            f"Each one prompts for the value. See SETUP.md.",
            file=sys.stderr,
        )
        return 1

    from google_auth_oauthlib.flow import InstalledAppFlow

    client_config = {
        "installed": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)

    set_env_value(settings.env_file, "GOOGLE_TOKEN_JSON", creds.to_json())

    print(f"Authorized. GOOGLE_TOKEN_JSON written to {settings.env_file}")
    print(f"Scopes granted: {', '.join(SCOPES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
