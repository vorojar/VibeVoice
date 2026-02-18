# Qwen3-TTS API Server

[中文](README.md)

A web application for Text-to-Speech powered by Alibaba's Qwen3-TTS model, featuring preset voices and voice cloning capabilities.

![Demo](demo_en.png)

> **Note**: Streaming audio output is not currently supported as the official SDK does not expose streaming interfaces yet.

---

## Features

- **Preset Speakers**: 9 built-in voices with emotion control
- **Voice Cloning**: Clone any voice from 3+ seconds of reference audio
- **Voice Library**: Save and manage cloned voices for reuse
- **Multilingual**: Support for 10 languages
- **Web UI**: Browser-based interface with recording support
- **REST API**: Easy integration into your applications

## Enhanced Features

1. **Sentence-by-Sentence Progress** - Long text is split into sentences and generated sequentially with real-time progress display, showing which sentence is currently being processed
2. **Stop Generation** - Cancel ongoing generation at any time; the system stops at the next sentence boundary
3. **Subtitle Generation** - Automatically generates SRT/VTT subtitle files synchronized with audio timing, perfect for video production
4. **Auto Language Detection** - Automatically detects input language (Chinese/English/Japanese/Korean) and sets the language selector accordingly
5. **Voice Prompt Caching** - Voice library entries cache their voice prompts to disk, eliminating repeated prompt extraction and significantly speeding up subsequent generations
6. **MP3 Export** - Convert and download audio as MP3 directly in the browser (in addition to WAV)
7. **Voice Design** - Create custom voices by describing characteristics in natural language (e.g., "deep male voice with a warm tone"), with automatic cross-sentence timbre consistency
8. **Sentence Editor** - After generation, select, edit, regenerate, delete or insert sentences individually, with full progress feedback during insertion

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

Server runs at http://localhost:8000

### Web Interface

Open http://localhost:8000 in your browser to access the web UI.

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
curl "http://localhost:8000/tts?text=Hello&speaker=aiden&language=English" -o output.wav

# With emotion instruction
curl "http://localhost:8000/tts?text=Hello&speaker=aiden&language=English&instruct=say it happily" -o output.wav
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
curl -X POST "http://localhost:8000/clone" \
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
curl http://localhost:8000/voices

# Save a voice
curl -X POST "http://localhost:8000/voices/save" \
  -F "name=MyVoice" \
  -F "language=English" \
  -F "audio=@reference.wav"

# Use saved voice
curl -X POST "http://localhost:8000/voices/{voice_id}/tts" \
  -F "text=Hello world" \
  -o output.wav

# Delete voice
curl -X DELETE "http://localhost:8000/voices/{voice_id}"
```

### Other Endpoints

```bash
# Get available speakers
curl http://localhost:8000/speakers

# Get supported languages
curl http://localhost:8000/languages
```

---

## Project Structure

```
├── api_server.py      # FastAPI server
├── index.html         # Web UI
├── test_qwen_tts.py   # Test script
├── models/            # Model files (not in repo)
└── saved_voices/      # Saved cloned voices
```

---

## Changelog

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

## License

This project uses the Qwen3-TTS model. Please refer to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for model license terms.
