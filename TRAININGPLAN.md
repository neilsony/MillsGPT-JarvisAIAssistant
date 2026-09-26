# Phase 0 — Getting to a Dmills voice, step by step

## First: what are we actually doing?

You want the bot to sound like Darrick. To do that, an AI voice model needs to
hear **a lot of Darrick talking, and nobody else.** Maybe 45 minutes of it.

That sounds easy — there are 500 episodes. But every episode has four friends
talking over each other, finishing each other's sentences, laughing at the same
time. There is no "Darrick track." It's all one mixed-down audio file.

So the whole of Phase 0 is one job: **pull Darrick's voice out of the pile.**

### What diarization is

"Diarization" is an ugly word for a simple idea: *figuring out who spoke when.*

Picture a recording of four friends at dinner. A **transcript** tells you what
was said. **Diarization** tells you who said it, and when — it draws a timeline:

```
0:00 ──────── 0:12   Person A
0:12 ── 0:20         Person B
0:20 ──── 0:31       Person A
0:31 ─ 0:33          Person C   ← talking over A
```

Here's the important limitation: **it has no idea who these people are.** It can
hear that there are four distinct voices, but it can't name them. So it calls
them `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`, `SPEAKER_03`.

Worse, **the numbers are random for each episode.** `SPEAKER_00` might be Kenny
in one episode and Darrick in the next. There is no consistency to rely on.

### Labeling the Speakers

Since the speaker numbers are random per episode, we can't just say "Darrick is
SPEAKER_02" once and be done. We'd have to check all 500 episodes by hand.

Instead we use a **voice embedding** — think of it as a voice fingerprint. It's a
list of a few hundred numbers describing what makes a voice distinctive: pitch,
resonance, the shape of the vowels, the rasp. Two clips of the *same person*
produce very similar number-lists. Two different people produce different ones.
It's face recognition, for voices.

So the plan becomes:

1. You identify Darrick **once**, by ear, in **one** episode.
2. The computer builds his fingerprint from those clips.
3. For every other episode, the computer compares each of the four speakers to
   that fingerprint and picks the closest match. No more listening required.

That's why there's exactly one manual step in this whole pipeline, and it's step 6.

### The safety rail

Four close friends, recording on similar microphones, in the same room, will
sometimes score *almost identically* against the fingerprint. When that happens
the honest answer is "I can't tell," not a coin flip.

So the pipeline refuses to guess. Episodes where the top two speakers are too
close get marked `AMBIGUOUS` and skipped. You'll lose a few episodes. That's
fine — there are 500, and the alternative is silently poisoning your training
set with 60 seconds of Kenny that you won't discover until you hear the finished
voice weeks later.

### The whole thing, end to end

```
  1. Download episodes                     →  data/audio/*.wav
  2. Diarize (who spoke when)              →  data/turns/*.json
  3. YOU listen, name Darrick once         ←  the only manual step
  4. Build his voice fingerprint           →  voices/dmills/reference.npy
  5. Auto-find him in every episode        →  data/labels.json
  6. Cut out clean solo Darrick audio      →  voices/dmills/clips/
  7. Feed that to a voice cloning model    →  he can now say anything
```

Steps 1-6 are the commands below. Step 7 is the bake-off at the end.

---


Everything here runs on your MacBook. No hardware, no purchases, no API bills.
The goal is a 45-minute training set of Darrick talking alone, and three
synthesized voices to compare by ear.

Budget roughly one evening of setup and one overnight run.

---

## 1. System tools (~5 min)

```bash
brew install ffmpeg yt-dlp
```

`ffmpeg` does the audio conversion; `yt-dlp` pulls episodes. Both are required
before anything downloads.

## 2. Python environment (~10 min, ~2 GB download)

```bash
cd ~/Desktop/MyJarvisAIBot
python3 -m venv .venv           # already done, skip if .venv exists
source .venv/bin/activate
pip install -e '.[pipeline]'
```

That pulls PyTorch and pyannote. It is the big one — go make coffee.

## 3. HuggingFace token and gated models (~5 min) ← **the usual blocker**

pyannote's models are free but gated: you must accept terms per model, signed
in, or you get an unhelpful 401 much later.

1. Make a token at <https://huggingface.co/settings/tokens> (read access is enough).
2. While **signed in**, visit each of these and click *Agree*:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/embedding>

   All three. The first one silently depends on the other two.
3. Put the token in `.env`, encrypted:

```bash
cp .env.example .env
npx @dotenvx/dotenvx set HF_TOKEN   # prompts for it; the first `set` also makes the keypair
```

Check it landed:

```bash
python -m pipeline status
```

## 4. Fetch episodes (~20 min for 5)

**Start with 5, not 20.** Prove the pipeline works before committing hours.

```bash
python -m pipeline list --limit 10      # see what's available
python -m pipeline fetch --limit 5
```

Audio lands in `data/audio/` as 24 kHz mono WAV — higher than diarization needs,
because the TTS engines want it and re-downloading later is painful.

## 5. Diarize (~10-20 min *per episode*) ← **run this overnight**

```bash
python -m pipeline diarize
```

This is the slow step: pyannote is largely CPU-bound on Apple Silicon, so a
90-minute episode takes 10-20 minutes. Five episodes is an hour or two; twenty
is an overnight job. **Results are cached** — interrupting it is safe, and
re-running picks up where it stopped.

You should see 4 speakers per episode. If you see 2 or 7, something's off — tell
me before continuing.

## 6. Find Darrick by ear (~5 min)

The diarizer knows there are four people but not who they are, and its labels
are meaningless across episodes. So listen:

```bash
python -m pipeline sample <episode-id>     # the stem from data/audio/
open data/samples/<episode-id>/
```

You get three clips per speaker. Play them, find Darrick, note the label. Then:

```bash
python -m pipeline reference --from-episode <episode-id> --speaker SPEAKER_02
```

That builds his voice fingerprint from eight clean turns. **This is the one step
only you can do** — everything downstream trusts it, so be sure before moving on.

## 7. Identify him everywhere else (~1 min per episode)

```bash
python -m pipeline identify
```

Each episode reports a score and a margin. Episodes marked `AMBIGUOUS` are
skipped deliberately — four friends on similar mics sometimes score too close to
call, and guessing wrong poisons the training set. A few skips is normal. Most
episodes skipping means the reference in step 6 was wrong.

Sanity check the `share` column: Darrick should be roughly 20-30% of a four-host
show. 5% or 60% means the label is wrong.

## 8. Cut the training set (~2 min)

```bash
python -m pipeline voiceprint --minutes 45
```

Produces:

- `voices/dmills/clips/` — the fine-tuning set (Piper)
- `voices/dmills/reference.wav` — the single best clean turn, for zero-shot engines
- `voices/dmills/manifest.json` — what came from where

**Listen to `reference.wav` before going further.** If there's any second voice
in it, stop and fix step 6.

If you got far short of 45 minutes, fetch more episodes rather than loosening the
filters — quality matters far more than quantity here.

---

## 9. The bake-off

Now you find out what he sounds like. Two engines clone zero-shot from
`reference.wav` and need no training:

**NeuTTS Air** — built for on-device, this is the Pi-friendly candidate.
**Chatterbox-Turbo** — best expressiveness in blind tests, MPS-accelerated on your M4.

Generate the *same* paragraph through both. Use something with his cadence in it
— a real take, not "the quick brown fox." Then listen with fresh ears.

**Piper comes third and costs more.** It needs a fine-tuning run on a rented GPU
*and* transcripts for every clip (LJSpeech format), which means a Whisper pass we
haven't built yet. Only worth doing if a zero-shot engine sounds close enough to
be worth optimizing for the Pi.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `401` / `Could not load pyannote/...` | Step 3 — you missed one of the three gated models |
| `yt-dlp` fails on one video | Members-only or region-locked; the run skips it and continues |
| 2 or 7 speakers detected | Wrong episode type (solo show, clip compilation) — try another |
| Everything `AMBIGUOUS` | Reference from step 6 is wrong — redo it with a different episode |
| `reference.wav` has two voices | Same — the label was wrong |

---

## What's already verified vs. not

The pure logic — segment filtering, crosstalk rejection, speaker matching,
config parsing — is covered by 81 passing tests.

The wrappers around `yt-dlp` and `pyannote` are written against their documented
APIs but **have not been run**, because the 2 GB of ML dependencies aren't
installed yet. Expect to shake out a rough edge or two on the first real run.
Tell me what breaks and I'll fix it.

---

# Phase 1 — Talking to it (text only)

Built and ready. No microphone, no hardware — you type, it answers in character.

## 10. Install the brain dependencies

```bash
pip install -e '.[brain]'
```

## 11. Load the show into its memory

```bash
python -m pipeline ingest
```

This transcribes each episode (slow, cached), joins the words to the speakers
from step 5, relabels Darrick's turns as `DARRICK:`, chunks it, embeds it
locally, and stores it. After this the bot can actually quote the show.

You can run this before finishing the voice work — Canon and Voiceprint are
independent.

## 12. Talk to it

```bash
python -m brain.cli
```

```
DmillsGPT — text mode
  model: z-ai/glm-5.3-flash
  canon: 1843 chunks   memories: 0

you > what do the guys think about the thunder
mills > ...
```

It has three tools: search the show, remember something about you, recall it
later. **It has no filesystem or shell access** — none of Claude Code's built-in
tools are enabled, deliberately.

## 13. Tune the register

`brain/persona/register.md` is the file you'll edit most. It's deliberately thin:
the real speech patterns come from retrieved transcript, not from my description
of how he talks. Add to it only when you notice retrieval missing something —
a recurring bit, a verbal tic, a stance he always takes.

Edit, restart the CLI, talk to it again. That's the loop.

## 14. Spotify (optional, ~5 min)

Lets you say "play [X song]", "queue [X song]", "pause", "resume", "skip". Needs **Spotify Premium**:
the Web API refuses playback control on free accounts. The bot is a remote, not
a speaker, so Spotify has to be open on your Mac or phone.

1. At <https://developer.spotify.com/dashboard>, create an app. Tick **Web API**,
   and add this redirect URI **exactly** (Spotify rejects `localhost`):
   `http://127.0.0.1:8888/callback`
2. Add the app's Client ID and Client Secret to `.env`, encrypted:

```bash
npx @dotenvx/dotenvx set SPOTIFY_CLIENT_ID       # each one prompts for the value
npx @dotenvx/dotenvx set SPOTIFY_CLIENT_SECRET
# optionally, in .env: SPOTIFY_DEVICE_NAME=Web Player  (not a secret, so plaintext
# is fine; only to pick between several open devices)
pip install -e '.[spotify]'
python -m brain.authorize_spotify
```

That opens a browser once. After you approve, it writes `SPOTIFY_TOKEN_JSON`
into `.env`, encrypted. The bot refreshes that token itself from then on. Without the
client ID and secret, the bot runs as before with no Spotify tools.

3. **Give the bot its own speaker** (recommended). Without this, music goes to
   whatever device already has Spotify open, which you have to open by hand.
   With it, the bot starts a hidden Spotify player called **DmillsGPT** when it
   starts, and always plays there, through this machine's speakers:

```bash
brew install librespot
python -m brain.authorize_spotify --player   # sign in with the SAME Spotify account
```

That signs the player in once and saves its login to `data/librespot/`
(gitignored, readable by you only). It's the one Spotify credential outside
`.env`, because librespot only reads its own file format. If the player ever
fails to start, its log is `data/librespot/librespot.log`.
