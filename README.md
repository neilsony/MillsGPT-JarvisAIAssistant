# DmillsGPT

![MillsGPT web UI](docs/ui-screenshot.png)

A voice-driven personal AI assistant with a distinct personality — cloned from a real podcast host — that talks back in real time, remembers what you tell it, manages your Google Calendar, and answers questions grounded in a searchable corpus. It runs locally on macOS, speaks through a voice-clone TTS engine, and is progressively getting an Arduino-driven physical body.

```
              ┌──────────────────────────────────────────────────┐
              │                    body/                          │
   mic ──────►│  push-to-talk capture → Deepgram STT             │
              │        ↓ transcript                              │
              ├──────────────────────────────────────────────────┤
              │                    brain/                         │
              │  agent loop (OpenRouter · GLM 5.3 Flash)          │
              │  ├── search_show     ── Canon corpus (sqlite-vec) │
              │  ├── remember/recall ── long-term memory (SQLite) │
              │  ├── load_skill      ── on-demand instructions    │
              │  ├── calendar tools  ── Google Calendar (native)  │
              │  ├── spotify tools   ── own player (librespot)    │
              │  └── web_search      ── live grounding            │
              ├──────────────────────────────────────────────────┤
              │  TTS daemon (Chatterbox-Turbo, voice-cloned)      │
   speaker ◄──│  reply audio at 24 kHz                            │
              └──────────────────────────────────────────────────┘

   pipeline/  (offline, one-time + incremental)
   YouTube audio → pyannote diarization → speaker identification
                → clean voiceprint dataset → TTS reference voice
```

The system is three independent subsystems with deliberately narrow interfaces:

- **`pipeline/`** — offline corpus engineering. Turns messy four-speaker podcast audio into a verified clean single-speaker voice dataset.
- **`brain/`** — the agent. LLM tool-calling loop, persona, retrieval, memory, skills, calendar.
- **`body/`** — the voice loop. Records microphone audio, transcribes it, feeds the brain, speaks the reply.

---

## Features

### Agent (`brain/`)

- **Native tool-calling loop** against OpenRouter (default model: `z-ai/glm-5.3-flash`, OpenAI-compatible API). The loop lives in `brain/agent.py`: send messages → execute requested tools → feed results back, until the model answers.
- **Thirteen on-demand tools**, no context bloat — nothing is auto-injected:
  - `search_show` — semantic retrieval over the "Canon" store (transcribed, chunked, and embedded show corpus in SQLite + `sqlite-vec`).
  - `remember` / `recall` — persistent long-term memory, stored locally in SQLite.
  - `load_skill` — pulls the full text of a named skill file (markdown with frontmatter) only when a turn actually needs it. The agent's system prompt stays small; a spoken-reply bot cannot afford pages of procedure on every turn.
  - `get_calendar_events`, `create_calendar_event`, `update_calendar_event` — native Google Calendar integration (Google's own client libraries, not an MCP server). OAuth credentials live encrypted in `.env`; the access token auto-refreshes and re-encrypts itself in place.
  - `play_music`, `queue_track`, `pause_music`, `resume_music`, `skip_track` — Spotify playback control (optional; Premium required). Plays through the bot's own headless player, "DmillsGPT" (librespot, started with the Brain), or whatever device has Spotify open if that isn't set up. Playback only: no library or playlist edits.
  - `web_search` — live web grounding for anything the corpus can't answer.
- **Strict, safe tool surface** — the agent has no filesystem access, no shell, no editing. Everything is read-mostly: search, remember, recall, calendar, music playback.
- **Persona** — a tuned voice register (`brain/persona/register.md`) plus user-profile context (`profile/`), including a fandom file that shapes tone without ever being treated as a source of facts.
- **Skill discovery** — skills are markdown files with strict frontmatter; a malformed file raises rather than being silently ignored.

### Voice loop (`body/`)

- **Push-to-talk conversation**: press Enter to record, Enter to send. Audio → Deepgram (batch STT) → agent → Chatterbox-Turbo TTS → speakers.
- **Web UI** (`body/ui/`): the voice loop also serves a local page at `http://127.0.0.1:8765` (opened automatically; `--no-ui` skips it) — a big picture of Darrick over the NOTB art, a ring showing the Body state (listening, thinking, speaking), and a Talk button (or space) that works alongside Enter. The two pictures go in `data/ui/` as `darrick.*` and `notb.*` (gitignored); without them the page shows a placeholder.
- **Persistent TTS daemon** (`brain/tts/chatterbox_server.py`): Chatterbox-Turbo loads once into a Unix-socket daemon and stays warm, so per-utterance latency skips the ~30 s model load. The daemon auto-starts on first use.
- **Voice cloning**: all replies are synthesized in Darrick's voice, zero-shot from a single 33 s reference clip.
- **Hardware-ready**: `body/` includes pyserial support for the Arduino chassis under `arduino/`.

### Corpus pipeline (`pipeline/`)

The hard problem this project solves: extract one person's voice from multi-speaker podcast audio, with no clean stems to work from.

- **Download** episode audio via `yt-dlp` (24 kHz mono WAV, the format TTS engines want).
- **Diarization** via pyannote 3.1 — who spoke when, pinned to 4 speakers for the show, 1 for solo content.
- **Voice fingerprinting** — a human identifies the target speaker once, by ear, in one episode; an embedding-based reference (`reference.npy`) is built from eight clean turns.
- **Automatic identification everywhere else** — every other episode's speakers are matched against the fingerprint, scored, and **refused when ambiguous**. The pipeline never guesses: episodes where the top two candidates score too close are marked `AMBIGUOUS` and skipped, because a mislabeled clip poisons the training set silently.
- **Crosstalk rejection** — turns with overlapping speech are filtered out before they reach the voiceprint; the overlap fractions themselves are computed and used as ranking signals.
- **Voiceprint assembly** — cut clean solo clips into a training set with a full provenance manifest (`voices/dmills/manifest.json`: which source video, exact timestamps, durations for every one of 205 clips).
- **Canon ingestion** — transcribe + chunk + embed show audio into the retrieval store.

---

## Current state

| Component | Status |
|---|---|
| Agent (tools, memory, skills, calendar, web) | ✅ Live, 426 tests passing |
| Voice loop (STT + TTS, push-to-talk) | ✅ Live |
| Voiceprint dataset | ✅ 205 clips / 29.6 min / 6 sources (4 solo videos + 2 confirmed show episodes) |
| TTS reference voice | ✅ `voices/dmills/reference.wav` (33 s clean turn) |
| Canon store | ✅ 1,686 chunks ingested |
| `.env` encryption (dotenvx) | ✅ Every credential encrypted at rest; decrypted in-process |

---

## CLI reference

All commands run from the repository root. Activate the main venv first: `source .venv/bin/activate`.

### Corpus pipeline — run these in order

```bash
python -m pipeline status                 # what's done and what's next
python -m pipeline list --limit 20        # list episodes without downloading
python -m pipeline fetch --limit 5        # download episode audio
python -m pipeline diarize                # who-spoke-when (slow, ~10–20 min/episode, cached & resumable)
python -m pipeline sample <episode-id>    # export clips per speaker, to identify by ear
python -m pipeline reference --from-episode <id> --speaker SPEAKER_02
                                          # build the voice fingerprint (one-time, human step)
python -m pipeline identify               # match Darrick in every episode; refuses ambiguous ones
python -m pipeline voiceprint --minutes 45
                                          # cut the clean training set + reference.wav + manifest
python -m pipeline ingest                 # transcribe + chunk + embed into the Canon store
python -m pipeline register-candidates    # surface speech-pattern candidates from Canon for review
```

Solo-channel variants (target speaker's own videos — single-speaker by construction):

```bash
python -m pipeline fetch-solo <channel-url> --limit 3
python -m pipeline diarize-solo           # fast: num_speakers=1, effectively VAD
python -m pipeline voiceprint-solo --minutes 45
python -m pipeline fetch-reference <url>  # one arbitrary clip as reference audio
```

### The assistant

```bash
python -m brain.cli                       # text mode — the dev loop for the agent
python -m body.voice_loop                 # voice mode — push-to-talk, plus the web UI
python -m body.voice_loop --no-ui         # voice mode, terminal only
python -m brain.authorize_google          # one-time Google OAuth consent → encrypted token into .env
python -m brain.tts.chatterbox_client "text"
                                          # TTS smoke test via the daemon
```

### Quality gates

```bash
pytest                                    # 426 tests
mypy brain body pipeline                  # strict mode
ruff check . && ruff format --check .
```

---

## Virtual environments

Three venvs, each isolating a ML stack whose dependency versions would otherwise fight. The TTS engine's stack is pinned separately from the main app so both stay installable and neither constrains the other's upgrades.

| Venv | Python | Purpose | Used by |
|---|---|---|---|
| `.venv` | 3.13 | Main app: agent, pipeline, voice loop, tests | `brain.cli`, `body.voice_loop`, `python -m pipeline`, `pytest` |
| `.venv-tts` | 3.13 | Chatterbox-Turbo TTS daemon | `brain/tts/chatterbox_server.py` |
| `.venv-pocket` | 3.13 | Kyutai Pocket TTS (evaluated, not adopted) | `pocket-tts` CLI |

```bash
# Main environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[pipeline,brain,google,body,dev]'

# TTS daemon (runs in its own venv; the client auto-spawns it)
python3 -m venv .venv-tts
.venv-tts/bin/pip install chatterbox-tts librosa soundfile resemble-perth

# TTS daemon lifecycle
pkill -f brain.tts.chatterbox_server      # stop it
rm -f /tmp/dmills-tts.sock                # clear a stale socket after a crash
```

The client (`ChatterboxTTS` in `brain/tts/chatterbox_client.py`) connects to `/tmp/dmills-tts.sock` and starts the daemon itself if it isn't running — no manual step in normal use.

---

## Dependencies

### Core (`pyproject.toml`, main venv)

| Package | Version | Role |
|---|---|---|
| pydantic | ≥ 2.9 | config/settings model |
| eciespy | 0.4.6 | decrypts/encrypts dotenvx values in `.env`, in-process |
| openai | 3.16.2 | OpenRouter client (OpenAI-compatible) |
| fastapi / uvicorn | 0.141.1 / 0.53.0 | service layer |
| sqlite-vec | 0.1.9 | vector search inside SQLite |
| sentence-transformers | 6.1.0 | embeddings for Canon + memory |
| deepgram-sdk | 7.9.0 | speech-to-text |
| google-api-python-client | 2.200.0 | Google Calendar |
| yt-dlp | 2026.8.19 | audio download |
| pyannote.audio | 4.0.7 | diarization + speaker embeddings |
| torch / torchaudio | 2.14.0 / 2.11.0 | ML runtime |
| faster-whisper | 1.2.1 | transcription for Canon ingest |
| sounddevice / soundfile | 0.5.6 / 0.13.1 | microphone capture, WAV I/O |
| webrtcvad / pyserial | 2.0.10+ / 3.5+ | voice-activity detection, Arduino serial link |
| pytest / pytest-cov / mypy / ruff | 9.1.1 / 7.1.0 / 2.3.1 / 0.16.5 | quality gates |

### TTS daemon (`.venv-tts`)

| Package | Version |
|---|---|
| chatterbox-tts | 0.1.7 |
| torch / torchaudio | 2.14.0 / 2.11.0 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| resemble-perth (watermarking) | 1.0.1 |

### System tools (Homebrew)

```bash
brew install ffmpeg yt-dlp
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"   # needed by some pipeline steps
brew install node                         # for `npx @dotenvx/dotenvx`, to add or change secrets
```

---

## Configuration

Everything lives in `.env` (gitignored; see `.env.example`), encrypted with [dotenvx](https://dotenvx.com). Real environment variables take precedence over file values.

| Variable | Required for | Notes |
|---|---|---|
| `HF_TOKEN` | `pipeline diarize` / `identify` | pyannote's three gated models need one-time terms acceptance on HuggingFace |
| `OPENROUTER_API_KEY` | agent (`brain.cli`, voice loop) | pay-as-you-go |
| `OPENROUTER_MODEL` | — | default `z-ai/glm-5.3-flash` |
| `DEEPGRAM_API_KEY` | voice loop | not needed in text mode |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_TOKEN_JSON` | calendar tools | desktop OAuth client; token auto-refreshes, re-encrypted, into `.env` |
| `DATA_DIR` / `VOICES_DIR` / `PROFILE_DIR` | — | optional path overrides |

**Encryption.** Every credential in `.env` is dotenvx ciphertext. `brain/config.py` decrypts values in-process as it loads them — no `dotenvx run` wrapper — with the private key from `DOTENV_PRIVATE_KEY`, `.env.keys`, or the OS secret store (macOS Keychain), in that order. Refreshed OAuth tokens are encrypted before they're written back, and a plaintext secret found in an encrypted `.env` stops the load. To add or change a secret without it ever touching disk — or your shell history — in plaintext:

```bash
npx @dotenvx/dotenvx set DEEPGRAM_API_KEY   # prompts for the value
```

Back up the private key (a password manager is fine): without it, nothing in `.env` can be decrypted.

---

## Data layout
gitignored, high level summary for tracking RAG: voice and real audio extraction data
```
data/
  audio/            downloaded episode WAVs (24 kHz mono)
  turns/            diarization results, one JSON per episode (cached, resumable)
  solo_audio/       target speaker's own-channel videos
  solo_turns/       their diarization (single-speaker)
  samples/          per-speaker clips for by-ear identification
  reference_audio/  arbitrary external reference clips
  labels.json       human-confirmed speaker labels (stronger than embedding matches)
  jarvis.db         SQLite: Canon chunks + long-term memories
voices/
  dmills/
    clips/          205 clean training clips
    manifest.json   provenance for every clip (source video, timestamps, duration)
    reference.wav   33 s zero-shot TTS reference
    reference.npy   speaker-embedding fingerprint
    bakeoff/        TTS engine comparison samples
```

---

## Testing & code standards

- **426 tests** (`pytest`), all passing — segment filtering, crosstalk rejection, speaker matching, persistence, tool schemas, prompt assembly, config parsing, and the agent loop are all covered without network access.
- **mypy strict mode** across all three packages, with targeted, documented exceptions for untyped third-party ML libraries.
- **ruff** with `E, F, I, UP, B, SIM` at 100 columns.

See `MASTER-PLAN.md` for the architecture rationale.

## To-do - from most to least urgent

- Setup Canon library from initial 5 episodes
- Improve prompt cache hit rate - Current = 29%, Goal = 75%. Try moving all dynamic system data to the end of the backend prompt structure. All static/rarely changing content inserted first.
- Add a Youtube Music tool
- Add two personas: goofy and serious. Use seperate register.md for each and allow toggle in UI.
- Revisit Pocket TTS to replace Chatterbox Turbo - Last attempt I wasn't happy with Pocket TTS voice quality, but it was 11x faster! Try to extract a better reference sample for the Pocket TTS voice model. Currently Chatterbox is way too slow but its quality is great - the TTS step during voice loop takes ~20 seconds on average or **85%** of total response time.
- If possible, new UI tool: let dmills pullup audio clips from its canon if prompted by the user. **Goal** - let this agent be used to search for certain clips or info from NOTB
- Deploy a basic version without my personal info to AWS
- add a basic arduino & microphone only body
