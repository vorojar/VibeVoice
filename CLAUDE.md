# CLAUDE.md
严格遵守，过程中任何回答，必须使用中文。
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-TTS web application - a Text-to-Speech service powered by Alibaba's Qwen3-TTS models. Three modes: preset voice synthesis (9 speakers + emotion control), voice cloning (from reference audio), and voice design (natural language voice description). Includes a voice library for saving/reusing cloned voices, sentence-by-sentence generation with progress, and a sentence editor for post-generation editing.

## Commands

### Run the server
```bash
python api_server.py
```
Server starts at http://localhost:8001 (port 8001).

### Test TTS generation
```bash
python test_qwen_tts.py
```
Generates `output_zh.wav` and `output_en.wav` test files.

### Download models
```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-1.7B-CustomVoice --local_dir ./models/Qwen3-TTS-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-1.7B-VoiceDesign --local_dir ./models/Qwen3-TTS-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-0.6B --local_dir ./models/Qwen3-TTS-0.6B
```

## Architecture

### Three Model System

Models are loaded **on-demand** via `POST /model/load/{model_type}` (background thread), not at startup. Status tracked in `model_status` dict (`unloaded` / `loading` / `loaded`). Auto-detects CUDA/MPS/CPU.

| Variable | Model | Purpose |
|----------|-------|---------|
| `model_custom` | `Qwen3-TTS-1.7B-CustomVoice` | 9 preset speakers with emotion control via `instruct` |
| `model_design` | `Qwen3-TTS-1.7B-VoiceDesign` | Voice from natural language description |
| `model_clone` | `Qwen3-TTS-0.6B` | Voice cloning from reference audio |

All models come from the `qwen_tts` package (`from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem`).

### Backend (api_server.py)

Single FastAPI file (~1636 lines). Key endpoint groups:

**Single-shot endpoints** — return raw WAV as StreamingResponse:
- `GET/POST /tts`, `POST /clone`, `POST /design`, `POST /voices/{id}/tts`

**SSE progress endpoints** — sentence-by-sentence generation with real-time progress:
- `GET /tts/progress`, `POST /clone/progress`, `GET /design/progress`, `GET /voices/{id}/tts/progress`

SSE event pattern: `{started}` → per-sentence `{progress}` → `{done}` with merged base64 WAV + `sentence_audios[]` array + `subtitles[]` + `stats`.

**Sentence operations:**
- `POST /regenerate` — single-sentence re-generation (all modes). Takes `sentence_text`, `mode`, `language`, plus mode-specific params (`speaker`/`instruct`/`voice_id`/`clone_prompt_id`).

**Voice library:** `/voices`, `/voices/save`, `/voices/{id}`, `/voices/{id}/tts`, `/voices/{id}/preview`, `/voices/{id}/cache`, `/voices/cache-all`

**Voice prompt caching:**
- `clone_session_prompts` dict: in-memory, keyed by UUID, 1hr expiry. Used by clone/design progress endpoints for cross-sentence voice consistency and by `/regenerate`.
- `saved_voice_prompts` dict: in-memory, persistent. Saved voices also cache `prompt.pt` on disk.

### Frontend

`index.html` is the HTML shell (~419 lines). Tailwind CSS via CDN with inline config in `<head>`. All JS in `static/js/`, CSS in `static/style.css`, served via FastAPI's `StaticFiles` mount at `/static`.

**JS load order** (all `<script defer>`, global scope, no ES modules):
1. `i18n.js` — zh/en translation strings, `t()`, `updateI18n()`, `toggleLanguage()`
2. `state.js` — all global state variables + IndexedDB session persistence (`saveSession`/`restoreSession`)
3. `audio.js` — WaveSurfer waveform, WAV/MP3 encode/decode, `mergeAllSentenceAudios()`, `rebuildAudioAndSubtitles()`, playback controls, subtitle download
4. `editor.js` — `switchMode()`, `splitTextToSentences()` (mirrors backend), progress view, sentence editor view (select/edit/delete/insert/instruct/undo)
5. `voice.js` — voice library UI + MediaRecorder recording
6. `generation.js` — `generate()` dispatcher, `generateWithProgress()` (GET/EventSource), `generateWithProgressPost()` (POST/fetch reader), `regenerateSentence()`, `stopGeneration()`, `ensureModelLoaded()`
7. `shortcuts.js` — keyboard shortcuts (Space=play, arrows=navigate, Enter=regenerate, P=preview, Delete=delete, Ctrl+Z=undo)
8. `main.js` — DOMContentLoaded entry point

**Two generation paths in frontend:**
- `generateWithProgress()` — uses GET + EventSource for preset and saved_voice modes
- `generateWithProgressPost()` — uses POST + fetch ReadableStream for clone and design modes (need to send file data)

### Key Data Flow

**Per-sentence arrays** (parallel arrays, maintained in sync):
- `sentenceTexts[]` — text of each sentence
- `sentenceAudios[]` — base64 WAV per sentence
- `sentenceInstructs[]` — per-sentence emotion instruction (preset mode only)

These are persisted to IndexedDB via `saveSession()` and restored on page load.

**Regeneration flow:** save to `undoStack` → POST `/regenerate` → replace array entry → `rebuildAudioAndSubtitles()` (client-side WAV merge) → `saveSession()`

### Voice Storage

Saved voices in `saved_voices/{timestamp}_{uuid8}/`:
- `meta.json` — name, language, ref_text, audio_file, created_at
- `reference.{ext}` — reference audio file
- `prompt.pt` — serialized `VoiceClonePromptItem` (cached on first use)

## Key Constants
- **SAMPLE_RATE**: 24000 Hz
- **SPEAKERS**: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

## Voice Cloning Modes
- With `ref_text`: Full ICL mode (higher quality)
- Without `ref_text`: `x_vector_only_mode=True` (faster, speaker embedding only)

## Design Mode Cross-Sentence Consistency
Multi-sentence text: first sentence uses `model_design`, then automatically switches to `model_clone` + first-sentence prompt for subsequent sentences. Single-sentence text uses pure `model_design` with no overhead.
