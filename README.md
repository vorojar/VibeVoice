# VibeVoice

[中文](README_zh.md) | [GitHub](https://github.com/vorojar/VibeVoice)

**Open-source AI audiobook studio** — a free, local alternative to ElevenLabs.

Powered by Qwen3-TTS models. Three voice creation modes, per-sentence granular control, LLM-powered smart character analysis, and multi-character mixed-voice generation. Runs 100% locally on your GPU — zero API costs, your data never leaves your device.

![Demo](demo_en.png)

---

## Why VibeVoice?

| Capability | VibeVoice | ElevenLabs |
|------------|:---------:|:----------:|
| Per-sentence emotion/voice control | ★★★★★ | ★★ |
| Voice creation modes | ★★★★★ Preset+Clone+Design | ★★★★ Clone+Preset |
| Smart character analysis | ★★★★★ LLM auto-analysis | ★★★ Manual tagging |
| Editing tools | ★★★★ Sentence editor+Preview | ★★ Basic editing |
| Voice quality | ★★★ | ★★★★★ |
| Real-time/streaming | ★★★ | ★★★★★ |
| Language coverage | ★★★★ 10 languages | ★★★★★ 32 languages |
| Cost | ★★★★★ Completely free | ★★ $5–$99/mo |
| Privacy | ★★★★★ 100% local | ★★ Cloud processing |

**Key advantages:**

- **Per-sentence granular control** — Set voice and emotion independently for each sentence. ElevenLabs only supports global settings. Critical for audiobooks where dialogue and narration need different tones
- **LLM smart analysis** — Qwen3-4B automatically identifies characters and emotions in text, one-click voice assignment. ElevenLabs requires manual paragraph-by-paragraph tagging
- **Voice design** — Create voices from natural language descriptions (e.g., "deep husky middle-aged male voice"), no reference audio needed. ElevenLabs doesn't offer this
- **Free + private** — Runs on your local GPU, no per-character billing, no usage limits, data never leaves your machine

---

## Features

### Three Voice Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Preset Speakers** | 9 built-in voices + emotion instruction control | Quick generation, no assets needed |
| **Voice Cloning** | Clone from 3s of reference audio | Replicate a specific voice |
| **Voice Design** | Create voice from natural language description | Design a new voice from scratch |

### Sentence Editor

- **Sentence preview** — Preview sentence splits before generation, edit text, adjust emotions, insert/delete, then generate
- **Per-sentence voice** — Each sentence can use a different voice (preset/library), interleave narration and character dialogue
- **Per-sentence emotion** — Each sentence gets its own emotion instruction (happy/sad/angry...)
- **Per-sentence editing** — Double-click to edit text, regenerate, delete, undo, single-sentence preview playback
- **Inter-sentence pause** — 0x–2x slider for real-time silence duration adjustment between sentences

### Smart Character Analysis

Powered by Qwen3-4B, automatically identifies characters and emotions in text. Character panel with one-click voice assignment, analysis results auto-fill per-sentence emotion instructions.

### Project Management

Sidebar project tree navigation, multi-project multi-chapter organization. Project-level character-voice mapping shared across chapters. IndexedDB persistence — refresh without losing work.

### More Features

- **Waveform visualizer** — WaveSurfer.js waveform player, highlights current sentence during playback
- **Subtitle export** — Auto-generates SRT/VTT subtitle files for video production
- **MP3 export** — In-browser conversion and download
- **Auto language detection** — Automatically sets language based on input text
- **Generation timer** — Real-time elapsed time display during generation
- **Keyboard shortcuts** — Space=play, arrows=navigate, Enter=regenerate, Ctrl+Z=undo
- **REST API** — Full HTTP API for integration
- **Voice prompt caching** — Disk-cached voice prompts for faster subsequent generation

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Preset Speakers

| Speaker | Language | Gender |
|---------|----------|--------|
| vivian | Chinese | Female |
| uncle_fu | Chinese | Male |
| aiden | English | Male |
| serena | English | Female |
| ono_anna | Japanese | Female |
| sohee | Korean | Female |
| dylan | - | Male |
| eric | - | Male |
| ryan | - | Male |

---

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended 8GB+ VRAM), or macOS Apple Silicon (M1/M2/M3)
- PyTorch with CUDA or MPS support

> **macOS Users**: Apple Silicon chips are supported via MPS backend with automatic detection using float16 precision. Actual compatibility depends on qwen-tts library's MPS support.

### Install Dependencies

```bash
pip install -U qwen-tts fastapi uvicorn python-multipart soundfile numpy torch
```

### Download Models

```bash
pip install -U modelscope

# CustomVoice model (preset speakers)
modelscope download --model Qwen/Qwen3-TTS-1.7B-CustomVoice --local_dir ./models/Qwen3-TTS-1.7B-CustomVoice

# VoiceDesign model (voice design)
modelscope download --model Qwen/Qwen3-TTS-1.7B-VoiceDesign --local_dir ./models/Qwen3-TTS-1.7B-VoiceDesign

# Base model (voice cloning)
modelscope download --model Qwen/Qwen3-TTS-0.6B --local_dir ./models/Qwen3-TTS-0.6B
```

---

## Usage

### Start Server

```bash
python api_server.py
```

Server runs at http://localhost:8001

### Web Interface

Open http://localhost:8001 in your browser to access the web UI.

---

## Performance Optimization

### Flash Attention (Recommended)

Installing Flash Attention can improve inference speed by approximately **50%**.

**Linux:**
```bash
pip install flash-attn --no-build-isolation
```

**Windows:**

Windows does not support source compilation. Use pre-built wheels from [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention/releases).

Example (Python 3.10 + PyTorch 2.9 + CUDA 12.8):
```bash
# Upgrade PyTorch first
pip install torch==2.9.0 torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install pre-built flash-attn
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

Verify installation: Server should no longer show `Warning: flash-attn is not installed` on startup.

### Further Optimization

To achieve higher performance (e.g., the official benchmark of 97ms/char):

| Solution | Expected Improvement | Notes |
|----------|---------------------|-------|
| Better GPU | 2-5x | A100/H100 vs consumer GPUs |
| vLLM Deployment | 2-3x | PagedAttention + continuous batching |
| TensorRT-LLM | 2-5x | NVIDIA official inference optimization |
| FP8 Quantization | 1.5-2x | Requires H100 |

> Consumer GPUs (RTX 40 series) + Flash Attention achieving ~1.4s/char is a reasonable expectation.

---

## API Reference

### TTS with Preset Speaker

```bash
# GET request
curl "http://localhost:8001/tts?text=Hello&speaker=aiden&language=English" -o output.wav

# With emotion instruction
curl "http://localhost:8001/tts?text=Hello&speaker=aiden&language=English&instruct=say it happily" -o output.wav
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| text | Text to synthesize | required |
| speaker | Speaker name | vivian |
| language | Language | Chinese |
| instruct | Emotion instruction | optional |

### Voice Cloning

```bash
curl -X POST "http://localhost:8001/clone" \
  -F "audio=@reference.wav" \
  -F "text=Hello world" \
  -F "language=English" \
  -F "ref_text=optional transcript of reference audio" \
  -o output.wav
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| audio | Reference audio file (3-10s) |
| text | Text to synthesize |
| language | Language |
| ref_text | Transcript of reference audio (optional, improves quality) |

### Voice Library

```bash
# List saved voices
curl http://localhost:8001/voices

# Save a voice
curl -X POST "http://localhost:8001/voices/save" \
  -F "name=MyVoice" \
  -F "language=English" \
  -F "audio=@reference.wav"

# Use saved voice
curl -X POST "http://localhost:8001/voices/{voice_id}/tts" \
  -F "text=Hello world" \
  -o output.wav

# Delete voice
curl -X DELETE "http://localhost:8001/voices/{voice_id}"
```

### Other Endpoints

```bash
# Get available speakers
curl http://localhost:8001/speakers

# Get supported languages
curl http://localhost:8001/languages
```

---

## Project Structure

```
├── api_server.py          # FastAPI server
├── index.html             # Web UI (HTML shell)
├── static/
│   ├── style.css          # Styles
│   └── js/
│       ├── i18n.js        # Chinese/English translations
│       ├── state.js       # Global state + IndexedDB project/chapter persistence
│       ├── audio.js       # Waveform player, WAV/MP3 encode/decode, audio merging
│       ├── editor.js      # Sentence editor, sentence preview, mode switching
│       ├── voice.js       # Voice library UI, recording, voice design
│       ├── generation.js  # Generation dispatch, SSE progress, regeneration
│       ├── shortcuts.js   # Keyboard shortcuts
│       └── main.js        # Entry point initialization
├── test_qwen_tts.py       # Test script
├── models/                # Model files (not in repo)
└── saved_voices/          # Saved cloned voices
```

---

## Changelog

### v0.4.0 (2025-02-19)

**Sentence Preview Mode**
- Preview sentence splits before generation, edit text, adjust emotion instructions, insert/delete sentences
- "Preview" → Edit → "Generate" three-step workflow, reducing wasted generations

**Per-Sentence Emotion & Voice Configuration**
- Each sentence can have its own emotion instruction (preset mode), independent of global emotion
- Each sentence can use a different voice (preset speaker / voice library), enabling mixed-voice generation
- Generation timer: real-time elapsed time display during generation (100ms refresh)

**LLM Smart Character/Emotion Analysis**
- Integrated Qwen3-4B model for automatic character and emotion recognition in text
- Character panel at the top of sentence editor with one-click voice assignment per character
- Analysis results auto-fill per-sentence emotion instructions

**Project/Chapter Management**
- Sidebar project tree navigation, multi-project multi-chapter organization
- Project-level character-voice mapping shared across chapters
- IndexedDB persistence (migrated from single-session to multi-project architecture), auto-migrates old data

**Voice Design Independent Mode**
- Voice design restored as independent 3rd tab (Preset | Library | Design)
- Preserves natural language expressiveness of voice descriptions, no longer downgraded to clone prompt

**Other Improvements**
- Paragraph boundary preservation: multi-paragraph text retains line break structure after sentence splitting
- Generation stats displayed in top-right status bar
- Keyboard shortcuts: Space=play, arrows=navigate, Enter=regenerate, P=preview, Delete=delete, Ctrl+Z=undo

### v0.3.0 (2025-02-18)

**Waveform Visualizer**
- Integrated WaveSurfer.js replacing the simple progress bar, displaying audio waveform
- Highlights the current sentence during playback, dims already-played sentences

**Sentence Editor**
- Enter sentence editor view after generation, supporting per-sentence operations
- Click to select, double-click to edit text
- Per-sentence regeneration (with spinner feedback), undo support to revert to previous version
- Per-sentence deletion (with confirmation)
- Insert new sentences between existing ones (with placeholder row, spinner, disabled actions, auto-play on completion)
- Inter-sentence pause control (0x–2x slider, adjusts silence duration between sentences in real-time)
- Per-sentence preview playback (play button for individual sentence audio)

**Session Persistence (IndexedDB)**
- Generation results (per-sentence audio, text, subtitles, parameters) automatically saved to IndexedDB
- Auto-restores previous session on page refresh, no need to regenerate
- Preserves editing state including inter-sentence pause multiplier

**Voice Design Cross-Sentence Timbre Consistency**
- Multi-sentence generation uses the design model for the first sentence, then automatically switches to clone model + first-sentence prompt for subsequent sentences, ensuring consistent timbre
- Regeneration and sentence insertion also reuse the cached voice prompt for consistency
- Single-sentence text has no extra overhead, still uses pure design model
- Automatic fallback to design model when clone model is not loaded

**Backend**
- All 4 progress endpoints (tts/clone/design/saved_voice) return per-sentence base64 audio array `sentence_audios`
- New `POST /regenerate` endpoint for single-sentence regeneration (preset/clone/design/saved_voice modes)
- Clone session prompt caching mechanism (`clone_session_prompts`) with 1-hour auto-expiry

### v0.2.0

- Sentence-by-sentence progress display, stop generation, subtitle generation
- Voice prompt disk caching, MP3 export
- Voice design mode, auto language detection

### v0.1.0

- Preset speaker synthesis (9 voices + emotion control)
- Voice cloning (record/upload reference audio)
- Voice library management
- Multilingual support (10 languages)
- REST API, bilingual UI (Chinese/English)

---

## Roadmap

| Status | Feature | Description |
|:------:|---------|-------------|
| ✅ | Per-sentence voice/emotion control | Independent voice and emotion per sentence |
| ✅ | LLM smart character analysis | Qwen3-4B auto character and emotion recognition |
| ✅ | Project/chapter management | Multi-project, multi-chapter, sidebar tree |
| ✅ | Voice design independent mode | Create voices from natural language descriptions |
| 🔲 | Long text import | TXT/EPUB/Word file import |
| 🔲 | Whisper auto-transcription | Auto-transcribe reference audio text |
| 🔲 | Speed/pitch control | Per-sentence speed and pitch adjustment |
| 🔲 | Audio post-processing | Noise reduction, EQ, loudness normalization |
| 🔲 | Pronunciation dictionary | Custom pronunciation for names/terms |
| 🔲 | Real-time streaming | Pending upstream SDK support |

---

## License

VibeVoice uses Qwen3-TTS models. Please refer to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for model license terms.
