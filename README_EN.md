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

## License

This project uses the Qwen3-TTS model. Please refer to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for model license terms.
