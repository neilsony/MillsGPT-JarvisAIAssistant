"""Tests for .env parsing, encryption, and settings resolution.

A secrets file that silently drops a key produces a confusing 401 three layers
away, so malformed lines are loud and the real environment always wins. And
once the file is encrypted, nothing here may put plaintext back on disk.
"""

import pytest

from brain.config import (
    Settings,
    _secret_store_command,
    decrypt_value,
    encrypt_value,
    find_private_key,
    load_env_file,
    parse_env,
    set_env_value,
)

# A throwaway keypair and a value sealed to it by the real dotenvx CLI (2.30.0),
# so these tests prove compatibility with dotenvx itself rather than just this
# module agreeing with itself. The key opens nothing but the string below.
FIXTURE_PUBLIC_KEY = "0395e4f1b71fd4875daf88197705b7d91612c8a85c37931689578078f4012c5b69"
FIXTURE_PRIVATE_KEY = "b5e6a4a6fc11163b52928f97a96b9403de1b06d25d5a2605893e28ccfc747367"
FIXTURE_CIPHERTEXT = (
    "encrypted:BAaW3zCF6PJGBgsVq3595cr6UXTHeiSXjAkBwSpJrFkX3xVF6+D261ivO4phS9m2k2q/8BvunNx3XO9gYK"
    "Qg4dEu+Z4GdH6Q5s9IuyvyB6WSXLq1g6YRG1JX9EMNWGynY7BAK1oTyEZQNxyFv0PBFjkHgA=="
)
FIXTURE_PLAINTEXT = "fixture-secret ✓"
# A valid secp256k1 private key that isn't the fixture's, for the wrong-key cases.
OTHER_PRIVATE_KEY = "1" * 64


@pytest.fixture(autouse=True)
def isolated_key_sources(monkeypatch):
    """Keep every test away from this machine's real private keys.

    Otherwise the OS secret store would be queried for real, and an exported
    DOTENV_PRIVATE_KEY would quietly win over the fixture's.
    """
    monkeypatch.setattr("brain.config._secret_store_lookup", lambda public_key: None)
    monkeypatch.delenv("DOTENV_PRIVATE_KEY", raising=False)


def encrypted_env(tmp_path, body="", *, keys_file=True):
    """An `.env` set up for encryption with the fixture keypair, laid out the
    way dotenvx writes it, plus (by default) its `.env.keys`."""
    path = tmp_path / ".env"
    path.write_text(f'DOTENV_PUBLIC_KEY="{FIXTURE_PUBLIC_KEY}"\n\n# .env\n{body}')
    if keys_file:
        (tmp_path / ".env.keys").write_text(f"# .env\nDOTENV_PRIVATE_KEY={FIXTURE_PRIVATE_KEY}\n")
    return path


class TestParseEnv:
    def test_simple_assignment(self):
        assert parse_env("FOO=bar") == {"FOO": "bar"}

    def test_multiple_lines(self):
        assert parse_env("A=1\nB=2") == {"A": "1", "B": "2"}

    def test_ignores_blank_lines_and_comments(self):
        assert parse_env("\n# a comment\n\nFOO=bar\n") == {"FOO": "bar"}

    def test_strips_whitespace_around_key_and_value(self):
        assert parse_env("  FOO  =  bar  ") == {"FOO": "bar"}

    def test_allows_export_prefix(self):
        assert parse_env("export FOO=bar") == {"FOO": "bar"}

    def test_empty_value(self):
        assert parse_env("FOO=") == {"FOO": ""}

    def test_value_may_contain_equals(self):
        assert parse_env("TOKEN=abc=def==") == {"TOKEN": "abc=def=="}

    def test_double_quoted_value_preserves_spaces(self):
        assert parse_env('FOO="  bar baz  "') == {"FOO": "  bar baz  "}

    def test_single_quoted_value(self):
        assert parse_env("FOO='bar baz'") == {"FOO": "bar baz"}

    def test_quoted_value_keeps_hash(self):
        assert parse_env('KEY="sk-ant-#-not-a-comment"') == {"KEY": "sk-ant-#-not-a-comment"}

    def test_strips_inline_comment_from_unquoted_value(self):
        assert parse_env("FOO=bar  # trailing note") == {"FOO": "bar"}

    def test_hash_without_leading_space_is_part_of_the_value(self):
        # API keys legitimately contain '#'; only ' #' starts a comment.
        assert parse_env("FOO=bar#baz") == {"FOO": "bar#baz"}

    def test_last_duplicate_wins(self):
        assert parse_env("FOO=1\nFOO=2") == {"FOO": "2"}

    def test_malformed_line_raises_with_line_number(self):
        with pytest.raises(ValueError, match="line 2"):
            parse_env("FOO=bar\nthis is not valid\n")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="line 1"):
            parse_env("=value")


class TestLoadEnvFile:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env_file(tmp_path / "nope.env") == {}

    def test_reads_a_real_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=dg-test\n")
        assert load_env_file(p) == {"DEEPGRAM_API_KEY": "dg-test"}


class TestSetEnvValue:
    def test_creates_a_new_file_when_none_exists(self, tmp_path):
        p = tmp_path / ".env"
        set_env_value(p, "FOO", "bar")
        assert load_env_file(p) == {"FOO": "bar"}

    def test_appends_a_new_key_to_an_existing_file(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("EXISTING=1\n")
        set_env_value(p, "NEW", "2")
        assert load_env_file(p) == {"EXISTING": "1", "NEW": "2"}

    def test_replaces_an_existing_key_in_place(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("A=1\nTARGET=old\nB=2\n")
        set_env_value(p, "TARGET", "new")
        assert load_env_file(p) == {"A": "1", "TARGET": "new", "B": "2"}

    def test_does_not_disturb_comments_or_other_lines(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# a comment\nA=1\nTARGET=old\n")
        set_env_value(p, "TARGET", "new")
        assert "# a comment" in p.read_text()
        assert "A=1" in p.read_text()

    def test_value_with_embedded_double_quotes_survives_a_round_trip(self, tmp_path):
        # This is the actual use case: a JSON blob (the Google OAuth token).
        p = tmp_path / ".env"
        token = '{"access_token": "abc", "refresh_token": "xyz"}'
        set_env_value(p, "GOOGLE_TOKEN_JSON", token)
        assert load_env_file(p)["GOOGLE_TOKEN_JSON"] == token

    def test_only_the_matching_key_is_replaced_not_a_prefix_match(self, tmp_path):
        # FOO_BAR=1 must not be mistaken for a match on FOO.
        p = tmp_path / ".env"
        p.write_text("FOO_BAR=1\n")
        set_env_value(p, "FOO", "2")
        got = load_env_file(p)
        assert got["FOO_BAR"] == "1"
        assert got["FOO"] == "2"

    def test_rejects_a_value_containing_an_apostrophe(self, tmp_path):
        # Single-quoting is how embedded double quotes survive; a value with
        # its own apostrophe would close the quote early and corrupt the file.
        with pytest.raises(ValueError, match="apostrophe"):
            set_env_value(tmp_path / ".env", "KEY", "it's broken")


class TestSetEnvValueEncrypted:
    """A refreshed OAuth token goes to disk as ciphertext, never plaintext."""

    def test_encrypts_before_writing(self, tmp_path):
        p = encrypted_env(tmp_path)
        token = '{"access_token": "abc", "refresh_token": "xyz"}'
        set_env_value(p, "SPOTIFY_TOKEN_JSON", token)
        assert "access_token" not in p.read_text()
        assert decrypt_value(load_env_file(p)["SPOTIFY_TOKEN_JSON"], FIXTURE_PRIVATE_KEY) == token

    def test_replaces_an_encrypted_value_in_place(self, tmp_path):
        p = encrypted_env(tmp_path, f"A=1\nGOOGLE_TOKEN_JSON='{FIXTURE_CIPHERTEXT}'\nB=2\n")
        set_env_value(p, "GOOGLE_TOKEN_JSON", "refreshed")
        got = load_env_file(p)
        assert list(got) == ["DOTENV_PUBLIC_KEY", "A", "GOOGLE_TOKEN_JSON", "B"]
        assert decrypt_value(got["GOOGLE_TOKEN_JSON"], FIXTURE_PRIVATE_KEY) == "refreshed"

    def test_an_apostrophe_is_fine_once_encrypted(self, tmp_path):
        # Ciphertext is base64, so the single-quoting limit no longer applies.
        p = encrypted_env(tmp_path)
        set_env_value(p, "KEY", "it's fine")
        assert decrypt_value(load_env_file(p)["KEY"], FIXTURE_PRIVATE_KEY) == "it's fine"

    def test_a_refreshed_token_reads_back_through_settings(self, tmp_path, monkeypatch):
        # The real cycle: the token refreshes, the Brain restarts and loads it.
        monkeypatch.delenv("SPOTIFY_TOKEN_JSON", raising=False)
        p = encrypted_env(tmp_path)
        set_env_value(p, "SPOTIFY_TOKEN_JSON", '{"access_token": "new"}')
        assert Settings.load(env_file=p).spotify_token_json == '{"access_token": "new"}'


class TestEncryption:
    def test_decrypts_a_value_encrypted_by_dotenvx(self):
        assert decrypt_value(FIXTURE_CIPHERTEXT, FIXTURE_PRIVATE_KEY) == FIXTURE_PLAINTEXT

    def test_round_trip(self):
        token = '{"access_token": "abc", "refresh_token": "xyz"}'
        sealed = encrypt_value(token, FIXTURE_PUBLIC_KEY)
        assert sealed.startswith("encrypted:")
        assert "access_token" not in sealed
        assert decrypt_value(sealed, FIXTURE_PRIVATE_KEY) == token

    def test_the_same_value_never_encrypts_the_same_way_twice(self):
        # A fresh ephemeral key per value: two equal secrets can't be spotted
        # as equal by comparing their ciphertext.
        assert encrypt_value("x", FIXTURE_PUBLIC_KEY) != encrypt_value("x", FIXTURE_PUBLIC_KEY)

    def test_the_wrong_private_key_raises(self):
        with pytest.raises(ValueError):
            decrypt_value(FIXTURE_CIPHERTEXT, OTHER_PRIVATE_KEY)


class TestFindPrivateKey:
    def test_the_environment_comes_first(self, tmp_path, monkeypatch):
        (tmp_path / ".env.keys").write_text("DOTENV_PRIVATE_KEY=from-file\n")
        monkeypatch.setenv("DOTENV_PRIVATE_KEY", "from-env")
        assert find_private_key(tmp_path / ".env", FIXTURE_PUBLIC_KEY) == "from-env"

    def test_then_the_keys_file_beside_the_env_file(self, tmp_path):
        (tmp_path / ".env.keys").write_text("# .env\nDOTENV_PRIVATE_KEY=from-file\n")
        assert find_private_key(tmp_path / ".env", FIXTURE_PUBLIC_KEY) == "from-file"

    def test_then_the_os_secret_store_by_public_key(self, tmp_path, monkeypatch):
        asked = []

        def lookup(public_key):
            asked.append(public_key)
            return "from-store"

        monkeypatch.setattr("brain.config._secret_store_lookup", lookup)
        assert find_private_key(tmp_path / ".env", FIXTURE_PUBLIC_KEY) == "from-store"
        assert asked == [FIXTURE_PUBLIC_KEY]

    def test_none_when_there_is_no_key_anywhere(self, tmp_path):
        assert find_private_key(tmp_path / ".env", FIXTURE_PUBLIC_KEY) is None


class TestSecretStoreCommand:
    """Must match dotenvx's own lookup exactly, or a key it stored is invisible here."""

    def test_macos_keychain(self):
        assert _secret_store_command("03abc", "darwin") == [
            "/usr/bin/security", "find-generic-password", "-s", "dotenvx", "-a", "03abc", "-w",
        ]

    def test_linux_secret_service(self):
        assert _secret_store_command("03abc", "linux") == [
            "secret-tool", "lookup", "service", "dotenvx", "public-key", "03abc",
        ]

    def test_no_store_elsewhere(self):
        assert _secret_store_command("03abc", "win32") is None


class TestEncryptedSettings:
    def test_decrypts_in_process(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        p = encrypted_env(tmp_path, f'OPENROUTER_API_KEY="{FIXTURE_CIPHERTEXT}"\n')
        assert Settings.load(env_file=p).openrouter_api_key == FIXTURE_PLAINTEXT

    def test_the_real_environment_still_wins(self, tmp_path, monkeypatch):
        p = encrypted_env(tmp_path, f'OPENROUTER_API_KEY="{FIXTURE_CIPHERTEXT}"\n')
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-shell")
        assert Settings.load(env_file=p).openrouter_api_key == "sk-from-shell"

    def test_ciphertext_from_the_environment_is_opened_too(self, tmp_path, monkeypatch):
        # `dotenvx run` without its key passes values through still sealed;
        # that must never reach an API as if it were the key itself.
        p = encrypted_env(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", FIXTURE_CIPHERTEXT)
        assert Settings.load(env_file=p).openrouter_api_key == FIXTURE_PLAINTEXT

    def test_no_private_key_raises_instead_of_using_ciphertext(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        p = encrypted_env(tmp_path, f'OPENROUTER_API_KEY="{FIXTURE_CIPHERTEXT}"\n', keys_file=False)
        with pytest.raises(RuntimeError, match="no private key"):
            Settings.load(env_file=p)

    def test_the_wrong_private_key_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("DOTENV_PRIVATE_KEY", OTHER_PRIVATE_KEY)
        p = encrypted_env(tmp_path, f'OPENROUTER_API_KEY="{FIXTURE_CIPHERTEXT}"\n', keys_file=False)
        with pytest.raises(RuntimeError, match="couldn't be decrypted"):
            Settings.load(env_file=p)

    def test_a_load_with_nothing_encrypted_never_looks_for_a_key(self, tmp_path, monkeypatch):
        def lookup(public_key):
            raise AssertionError("looked up a private key it didn't need")

        monkeypatch.setattr("brain.config._secret_store_lookup", lookup)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        p = encrypted_env(tmp_path, "OPENROUTER_MODEL=z-ai/glm-5.3\n", keys_file=False)
        assert Settings.load(env_file=p).model == "z-ai/glm-5.3"


class TestPlaintextGuard:
    """Once .env is encrypted, a plaintext secret in it is a leak nobody would notice."""

    def test_a_secret_pasted_in_by_hand_is_refused(self, tmp_path):
        p = encrypted_env(tmp_path, "DEEPGRAM_API_KEY=dg-pasted-by-hand\n")
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            Settings.load(env_file=p)

    def test_the_fix_it_offers_is_set_not_encrypt(self, tmp_path):
        # `dotenvx encrypt` would cut an unquoted value at a bare '#';
        # `dotenvx set` takes the value exactly as typed.
        p = encrypted_env(tmp_path, "HF_TOKEN=hf_abc\n")
        with pytest.raises(RuntimeError, match="dotenvx set HF_TOKEN"):
            Settings.load(env_file=p)

    def test_a_key_this_module_has_never_heard_of_counts_as_a_secret(self, tmp_path):
        p = encrypted_env(tmp_path, "SOME_NEW_TOKEN=abc\n")
        with pytest.raises(RuntimeError, match="SOME_NEW_TOKEN"):
            Settings.load(env_file=p)

    def test_plain_settings_are_allowed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        p = encrypted_env(
            tmp_path, "OPENROUTER_MODEL=z-ai/glm-5.3\nSPOTIFY_DEVICE_NAME=Web Player\n"
        )
        assert Settings.load(env_file=p).model == "z-ai/glm-5.3"

    def test_empty_values_are_not_secrets(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        p = encrypted_env(tmp_path, "HF_TOKEN=\n")
        assert Settings.load(env_file=p).hf_token is None


class TestSettings:
    def test_real_environment_beats_the_env_file(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=from-file\n")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "from-shell")
        assert Settings.load(env_file=p).deepgram_api_key == "from-shell"

    def test_falls_back_to_the_env_file(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("DEEPGRAM_API_KEY=from-file\n")
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        assert Settings.load(env_file=p).deepgram_api_key == "from-file"

    def test_model_defaults_to_glm_5_3_flash(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        assert Settings.load(env_file=tmp_path / "none").model == "z-ai/glm-5.3-flash"

    def test_model_is_overridable_for_ab_testing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MODEL", "z-ai/glm-5.3")
        assert Settings.load(env_file=tmp_path / "none").model == "z-ai/glm-5.3"

    def test_missing_secret_is_none_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert Settings.load(env_file=tmp_path / "none").hf_token is None

    def test_require_raises_a_actionable_message(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        s = Settings.load(env_file=tmp_path / "none")
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            s.require("hf_token")

    def test_require_returns_the_value_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abc")
        assert Settings.load(env_file=tmp_path / "none").require("hf_token") == "hf_abc"
