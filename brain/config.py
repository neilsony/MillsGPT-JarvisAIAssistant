"""Settings and secrets for the whole project.

Secrets live in a gitignored `.env` at the repo root, encrypted with dotenvx:
each value is `encrypted:` plus ECIES ciphertext (secp256k1 + AES-256-GCM),
sealed to the public key written at the top of the file. Only ciphertext ever
touches disk:

- **Reading.** `Settings.load()` decrypts each value in-process, right where
  it's declared, with the private key from wherever dotenvx keeps it (see
  `find_private_key`). Plaintext exists only in this process's memory, and no
  `dotenvx run --` wrapper is needed.
- **Writing.** `set_env_value` — an OAuth token refreshing — encrypts with the
  file's public key *before* it writes, so a refreshed token is never on disk
  in plaintext, not even for a moment.
- **Guarding.** Once the file is encrypted, a secret that turns up in it as
  plaintext — pasted in by hand, or left behind by `dotenvx decrypt` — stops
  the load with the command that fixes it.

Real environment variables still win over the file, so `dotenvx run --`, a
shell export, or a service manager can hand a value in directly.

The model id lives here, alone, on purpose: swapping models while tuning the
register should be a one-line change, not a search-and-replace.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Settings",
    "decrypt_value",
    "encrypt_value",
    "find_private_key",
    "load_env_file",
    "parse_env",
    "set_env_value",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

# dotenvx's conventions, which everything below has to match exactly: the
# prefix on each encrypted value, the key names for a file called `.env`, the
# private-key file it falls back to, and the service it files private keys
# under in the OS secret store.
CIPHERTEXT_PREFIX = "encrypted:"
PUBLIC_KEY_NAME = "DOTENV_PUBLIC_KEY"
PRIVATE_KEY_NAME = "DOTENV_PRIVATE_KEY"
KEYS_FILE_NAME = ".env.keys"
SECRET_STORE_SERVICE = "dotenvx"

# The only things an encrypted `.env` may hold as plaintext: settings, not
# secrets. Any other key — including one this module has never heard of — is
# treated as a credential and must be ciphertext.
PLAINTEXT_ALLOWED = frozenset(
    {
        PUBLIC_KEY_NAME,
        "OPENROUTER_MODEL",
        "SPOTIFY_DEVICE_NAME",
        "DATA_DIR",
        "VOICES_DIR",
        "PROFILE_DIR",
    }
)

# Human-readable names for the error message when a secret is missing.
_ENV_NAMES = {
    "deepgram_api_key": "DEEPGRAM_API_KEY",
    "hf_token": "HF_TOKEN",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "google_client_id": "GOOGLE_CLIENT_ID",
    "google_client_secret": "GOOGLE_CLIENT_SECRET",
    "spotify_client_id": "SPOTIFY_CLIENT_ID",
    "spotify_client_secret": "SPOTIFY_CLIENT_SECRET",
}


def parse_env(text: str) -> dict[str, str]:
    """Parse dotenv-format `text`.

    Deliberately strict: a line that isn't blank, a comment, or a `KEY=value`
    raises rather than being skipped. In a secrets file a typo'd line is a bug
    you want at parse time, not as a 401 from a service three layers down.

    Quoting rules follow the usual dotenv convention — quoted values are taken
    literally (so a `#` inside an API key survives), unquoted values have a
    ` #` trailing comment stripped.
    """
    values: dict[str, str] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"malformed .env line {lineno}: expected KEY=value, got {raw!r}")

        key = key.strip()
        if not key:
            raise ValueError(f"malformed .env line {lineno}: empty key in {raw!r}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            # Only ' #' opens a comment — bare '#' is legal inside a token.
            comment = value.find(" #")
            if comment != -1:
                value = value[:comment].rstrip()

        values[key] = value

    return values


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file, or return {} when it doesn't exist.

    A missing `.env` is normal — everything in it may be set in the shell instead.
    """
    if not path.is_file():
        return {}
    return parse_env(path.read_text(encoding="utf-8"))


def encrypt_value(value: str, public_key: str) -> str:
    """Seal `value` to `public_key`, in exactly the form `dotenvx encrypt` writes.

    Only the public key is needed, so writing a secret into `.env` never
    requires being able to read one back out.
    """
    import ecies

    sealed = ecies.encrypt(public_key, value.encode("utf-8"))
    return CIPHERTEXT_PREFIX + base64.b64encode(sealed).decode("ascii")


def decrypt_value(value: str, private_key: str) -> str:
    """Open a dotenvx `encrypted:` value. A wrong key raises ValueError."""
    import ecies

    sealed = base64.b64decode(value.removeprefix(CIPHERTEXT_PREFIX), validate=True)
    return ecies.decrypt(private_key, sealed).decode("utf-8")


def find_private_key(env_file: Path, public_key: str | None) -> str | None:
    """The private key that opens `env_file`, from wherever dotenvx keeps it.

    Checked in dotenvx's own order:

    1. `DOTENV_PRIVATE_KEY` in the environment — how a headless machine (the
       Pi) gets handed the key by whatever starts the Brain.
    2. `.env.keys` beside the file — dotenvx's on-disk fallback.
    3. The OS secret store (macOS Keychain, Linux Secret Service), filed under
       the file's public key — dotenvx 2's default, and the only one that
       leaves no plaintext key on disk at all.
    """
    from_environment = os.environ.get(PRIVATE_KEY_NAME)
    if from_environment:
        return from_environment
    from_keys_file = load_env_file(env_file.parent / KEYS_FILE_NAME).get(PRIVATE_KEY_NAME)
    if from_keys_file:
        return from_keys_file
    return _secret_store_lookup(public_key) if public_key else None


def _secret_store_command(public_key: str, platform: str) -> list[str] | None:
    """How dotenvx itself reads a private key back out of the OS secret store:
    the same binaries and service name, keyed by the file's public key."""
    if platform == "darwin":
        return [
            "/usr/bin/security", "find-generic-password",
            "-s", SECRET_STORE_SERVICE, "-a", public_key, "-w",
        ]
    if platform.startswith("linux"):
        return ["secret-tool", "lookup", "service", SECRET_STORE_SERVICE, "public-key", public_key]
    return None


def _secret_store_lookup(public_key: str) -> str | None:
    """The private key filed under `public_key` in the OS secret store, if any.

    Every failure — no store on this machine, no entry, a locked keychain —
    just means "not here"; `Settings.load` then says where else it looked.
    """
    command = _secret_store_command(public_key, sys.platform)
    if command is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _refuse_plaintext_secrets(env_file: Path, values: dict[str, str]) -> None:
    """Stop if an encrypted `.env` is holding a secret in plaintext.

    That's a key pasted in by hand, or a file decrypted to edit and never
    re-encrypted — both quiet, and both exactly what encrypting the file was
    meant to rule out. Refusing to start is the loud version.

    The fix it offers is `dotenvx set`, not `dotenvx encrypt`: encrypt re-reads
    the line with dotenvx's own parser, which ends an unquoted value at a bare
    `#` and would silently seal a truncated token. `set` takes the value as-is.
    """
    exposed = sorted(
        name
        for name, value in values.items()
        if value and name not in PLAINTEXT_ALLOWED and not value.startswith(CIPHERTEXT_PREFIX)
    )
    if exposed:
        commands = "\n".join(f"  npx @dotenvx/dotenvx set {name} -f {env_file}" for name in exposed)
        raise RuntimeError(
            f"{env_file} is encrypted, but has plaintext secrets: {', '.join(exposed)}.\n"
            f"Re-enter each one encrypted (it prompts, and replaces the plaintext line):\n"
            f"{commands}"
        )


def set_env_value(path: Path, key: str, value: str) -> None:
    """Write one `KEY=value` line into a dotenv file, touching nothing else.

    For values that legitimately change at runtime and must persist — right
    now, only the Google and Spotify OAuth tokens after they refresh. Everything
    else in the file (comments, blank lines, other keys) is preserved
    byte-for-byte; only the matching `KEY=` line is replaced, or appended if
    absent.

    In an encrypted file the value is encrypted with the file's public key
    *before* it's written, so a refreshed token never sits on disk in
    plaintext — not even between a write and a re-encrypt.

    A file that isn't encrypted yet gets the value single-quoted: the token is
    a JSON blob full of double quotes, and single quotes are the one form
    `parse_env` reads back literally without those inner quotes closing the
    value early.
    """
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    public_key = parse_env(text).get(PUBLIC_KEY_NAME)

    if public_key:
        new_line = f'{key}="{encrypt_value(value, public_key)}"'
    elif "'" in value:
        raise ValueError("set_env_value cannot safely quote a value containing an apostrophe")
    else:
        new_line = f"{key}='{value}'"

    lines = text.splitlines()
    prefix = f"{key}="

    for i, raw in enumerate(lines):
        stripped = raw.strip().removeprefix("export ").lstrip()
        if stripped.startswith(prefix):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Settings:
    """Resolved configuration. Build with `Settings.load()`."""

    model: str
    deepgram_api_key: str | None
    hf_token: str | None
    openrouter_api_key: str | None
    data_dir: Path
    voices_dir: Path
    profile_dir: Path
    # Google OAuth — all three live in .env, encrypted like everything else,
    # never as a client-secrets JSON file or a separate token file. `env_file`
    # is kept so the token can be re-encrypted in place after it refreshes;
    # see brain/tools/calendar.py.
    google_client_id: str | None
    google_client_secret: str | None
    google_token_json: str | None
    # Spotify OAuth — same arrangement as Google; see brain/tools/spotify.py.
    # `spotify_device_name` is an optional tiebreaker when several devices
    # have Spotify open and none is active (a substring of its name).
    spotify_client_id: str | None
    spotify_client_secret: str | None
    spotify_token_json: str | None
    spotify_device_name: str | None
    env_file: Path

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        resolved_env_file = env_file if env_file is not None else DEFAULT_ENV_FILE
        file_values = load_env_file(resolved_env_file)
        public_key = file_values.get(PUBLIC_KEY_NAME) or None
        if public_key is not None:
            _refuse_plaintext_secrets(resolved_env_file, file_values)

        private_key: str | None = None

        def reveal(name: str, sealed: str) -> str:
            # The key is looked up once, and only if something is actually
            # encrypted — a settings-only load never touches the secret store.
            nonlocal private_key
            if private_key is None:
                private_key = find_private_key(resolved_env_file, public_key)
            if private_key is None:
                raise RuntimeError(
                    f"{name} is encrypted, but no private key was found to open it.\n"
                    f"dotenvx keeps it in one of: ${PRIVATE_KEY_NAME}, "
                    f"{resolved_env_file.parent / KEYS_FILE_NAME}, or the OS secret store "
                    f"(macOS Keychain / Linux Secret Service) under this .env's public key."
                )
            try:
                return decrypt_value(sealed, private_key)
            except ValueError as exc:
                raise RuntimeError(
                    f"{name} couldn't be decrypted: either the private key found isn't "
                    f"the one for {resolved_env_file}, or the value itself is damaged."
                ) from exc

        def get(name: str, default: str | None = None) -> str | None:
            # Real environment wins; the file is the fallback. Ciphertext is
            # opened wherever it came from: `dotenvx run` without its key
            # passes values through still sealed, and a sealed value must
            # never reach an API as if it were the key itself.
            value = os.environ.get(name) or file_values.get(name)
            if value and value.startswith(CIPHERTEXT_PREFIX):
                value = reveal(name, value)
            return value or default

        data_dir = Path(get("DATA_DIR") or REPO_ROOT / "data")

        return cls(
            # GLM 5.3 Flash via OpenRouter — see MASTER-PLAN.md § Running cost.
            # Override with OPENROUTER_MODEL to A/B during register tuning.
            model=get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash") or "z-ai/glm-5.3-flash",
            deepgram_api_key=get("DEEPGRAM_API_KEY"),
            hf_token=get("HF_TOKEN"),
            openrouter_api_key=get("OPENROUTER_API_KEY"),
            data_dir=data_dir,
            voices_dir=Path(get("VOICES_DIR") or REPO_ROOT / "voices"),
            profile_dir=Path(get("PROFILE_DIR") or REPO_ROOT / "profile"),
            google_client_id=get("GOOGLE_CLIENT_ID"),
            google_client_secret=get("GOOGLE_CLIENT_SECRET"),
            google_token_json=get("GOOGLE_TOKEN_JSON"),
            spotify_client_id=get("SPOTIFY_CLIENT_ID"),
            spotify_client_secret=get("SPOTIFY_CLIENT_SECRET"),
            spotify_token_json=get("SPOTIFY_TOKEN_JSON"),
            spotify_device_name=get("SPOTIFY_DEVICE_NAME"),
            env_file=resolved_env_file,
        )

    def require(self, field: str) -> str:
        """Return a secret, or explain exactly how to set it.

        Used at the point of use rather than at load time, so running the corpus
        pipeline doesn't demand a Deepgram key it will never touch.
        """
        value = getattr(self, field)
        if not value:
            name = _ENV_NAMES.get(field, field.upper())
            raise RuntimeError(
                f"{name} is not set. Add it, encrypted, with:\n"
                f"  npx @dotenvx/dotenvx set {name} -f {self.env_file}\n"
                f"(it prompts for the value, so it never lands in shell history), "
                f"or export it in your shell. See SETUP.md."
            )
        return str(value)
