"""One-time Spotify authorization.

Run once, interactively. It opens a browser, you consent, and it writes
`SPOTIFY_TOKEN_JSON` straight into your `.env`, encrypted — no separate token
file. The bot re-reads it (and re-encrypts it in place after each refresh)
from there onward.

    python -m brain.authorize_spotify            # the Web API token (controls playback)
    python -m brain.authorize_spotify --player   # the bot's own player (librespot)

The two are separate sign-ins because they're separate things: the token lets
the bot *tell* a device what to play; the player *is* a device. Sign both in
to the same Spotify account, or the player won't appear in that account's
device list.

Kept out of the agent path on purpose: a bot that can pop a browser consent
screen mid-conversation is a bot that hangs waiting for a click nobody saw.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import webbrowser

from brain.config import Settings
from brain.spotify_player import (
    CREDENTIALS_FILE,
    PLAYER_NAME,
    player_cache_dir,
    player_command,
    secure_cache_dir,
)
from brain.tools.spotify import REDIRECT_URI, SCOPES, EnvTokenStore, make_auth_manager

_AUTH_URL = re.compile(r"https://accounts\.spotify\.com/\S+")


def authorize_api(settings: Settings) -> int:
    if not settings.spotify_client_id or not settings.spotify_client_secret:
        print(
            f"SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET aren't set in {settings.env_file}.\n\n"
            f"At developer.spotify.com/dashboard: create an app, tick 'Web API', and add\n"
            f"the redirect URI {REDIRECT_URI} exactly. Then add its Client ID and Client\n"
            f"Secret to your .env, encrypted:\n\n"
            f"  npx @dotenvx/dotenvx set SPOTIFY_CLIENT_ID\n"
            f"  npx @dotenvx/dotenvx set SPOTIFY_CLIENT_SECRET\n\n"
            f"Each one prompts for the value. See SETUP.md.",
            file=sys.stderr,
        )
        return 1

    # Start from no token so spotipy always runs the browser flow; the store
    # writes whatever comes back into .env.
    store = EnvTokenStore(settings.env_file, None)
    auth_manager = make_auth_manager(
        settings.spotify_client_id, settings.spotify_client_secret, store, open_browser=True
    )
    auth_manager.get_access_token(as_dict=True, check_cache=False)

    if store.get() is None:
        print("Spotify didn't return a token. Try again.", file=sys.stderr)
        return 1

    print(f"Authorized. SPOTIFY_TOKEN_JSON written to {settings.env_file}")
    print(f"Scopes granted: {', '.join(SCOPES)}")
    return 0


def authorize_player(settings: Settings) -> int:
    binary = shutil.which("librespot")
    if binary is None:
        print("librespot isn't installed. Run: brew install librespot", file=sys.stderr)
        return 1

    cache_dir = player_cache_dir(settings.data_dir)
    secure_cache_dir(cache_dir)
    credentials = cache_dir / CREDENTIALS_FILE

    # librespot's own OAuth flow: it prints a sign-in URL, catches the redirect
    # on a local port, and writes credentials.json. It then keeps running as a
    # player, so stop it as soon as the credentials land.
    process = subprocess.Popen(
        [*player_command(binary, cache_dir), "--enable-oauth"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    opened = False
    try:
        assert process.stdout is not None
        for line in process.stdout:
            match = _AUTH_URL.search(line)
            if match and not opened:
                print(f"Opening the Spotify sign-in page. If it doesn't open, visit:\n"
                      f"  {match.group(0)}\n")
                webbrowser.open(match.group(0))
                opened = True
            if credentials.is_file():
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    if not credentials.is_file():
        print("librespot exited without saving credentials. Try again.", file=sys.stderr)
        return 1

    credentials.chmod(0o600)
    print(f"Player signed in. The bot will play through '{PLAYER_NAME}' from now on.")
    print(f"Credentials: {credentials} (gitignored, readable by you only)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m brain.authorize_spotify")
    parser.add_argument(
        "--player",
        action="store_true",
        help=f"sign in the bot's own Spotify player ({PLAYER_NAME}) instead of the Web API",
    )
    args = parser.parse_args()
    settings = Settings.load()
    return authorize_player(settings) if args.player else authorize_api(settings)


if __name__ == "__main__":
    raise SystemExit(main())
