# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-TTS web application - a Text-to-Speech service powered by Alibaba's Qwen3-TTS model. Provides preset voice synthesis, voice cloning, and a voice library for saving cloned voices.

## Commands

### Run the server
```bash
python api_server.py
```
Server starts at http://localhost:8000 with web UI at root path.

### Test TTS generation
```bash
python test_qwen_tts.py
```
Generates `output_zh.wav` and `output_en.wav` test files.

### Download models (if not present)
```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-0.6B-CustomVoice --local_dir ./models/Qwen3-TTS-0.6B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-0.6B --local_dir ./models/Qwen3-TTS-0.6B
```

## Architecture

### Dual Model System
- **model_custom** (`Qwen3-TTS-0.6B-CustomVoice`): 9 preset speakers with emotion control
- **model_clone** (`Qwen3-TTS-0.6B`): Voice cloning from reference audio

Both models load at startup, run on CUDA with bfloat16 precision.

### API Endpoints (api_server.py)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/tts` | GET/POST | Preset speaker synthesis |
| `/clone` | POST | Voice cloning with uploaded audio |
| `/voices` | GET | List saved voices |
| `/voices/save` | POST | Save cloned voice to library |
| `/voices/{id}/tts` | POST | Generate with saved voice |
| `/voices/{id}` | DELETE | Delete saved voice |

### Voice Storage
Saved voices stored in `saved_voices/{timestamp}_{uuid}/`:
- `meta.json` - name, language, ref_text, created_at
- `reference.{ext}` - reference audio file

### Frontend (index.html)
Single-page app with Tailwind CSS. Two main panels:
1. **Custom Voice** - preset speakers + emotion instructions
2. **Voice Clone** - recording/upload + voice library management

Uses MediaRecorder API for browser recording, fetch() for API calls.

## Key Constants
- **SAMPLE_RATE**: 24000 Hz
- **SPEAKERS**: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian
- **LANGUAGES**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Voice Cloning Modes
- With `ref_text`: Full ICL mode (higher quality)
- Without `ref_text`: x_vector_only_mode=True (faster, speaker embedding only)
