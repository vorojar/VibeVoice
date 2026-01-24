# Qwen3-TTS API Server

A web application for Text-to-Speech powered by Alibaba's Qwen3-TTS model, featuring preset voices and voice cloning capabilities.

基于阿里巴巴 Qwen3-TTS 模型的语音合成 Web 应用，支持预设说话人和语音克隆功能。

![Demo](demo.png)

---

## Features / 功能特性

- **Preset Speakers / 预设说话人**: 9 built-in voices with emotion control (9 种内置声音，支持情感控制)
- **Voice Cloning / 语音克隆**: Clone any voice from 3+ seconds of reference audio (3秒以上参考音频即可克隆声音)
- **Voice Library / 声音库**: Save and manage cloned voices for reuse (保存和管理克隆的声音)
- **Multilingual / 多语言**: Support for 10 languages (支持10种语言)
- **Web UI / 网页界面**: Browser-based interface with recording support (支持浏览器录音的网页界面)
- **REST API**: Easy integration into your applications (便于集成到您的应用)

## Supported Languages / 支持的语言

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语

## Preset Speakers / 预设说话人

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

## Installation / 安装

### Requirements / 环境要求

- Python 3.8+
- CUDA-compatible GPU (recommended 8GB+ VRAM / 建议8GB以上显存)
- PyTorch with CUDA support

### Install Dependencies / 安装依赖

```bash
pip install -U qwen-tts fastapi uvicorn python-multipart soundfile numpy torch
```

### Download Models / 下载模型

```bash
pip install -U modelscope

# CustomVoice model (preset speakers) / 预设说话人模型
modelscope download --model Qwen/Qwen3-TTS-0.6B-CustomVoice --local_dir ./models/Qwen3-TTS-0.6B-CustomVoice

# Base model (voice cloning) / 语音克隆模型
modelscope download --model Qwen/Qwen3-TTS-0.6B --local_dir ./models/Qwen3-TTS-0.6B
```

---

## Usage / 使用方法

### Start Server / 启动服务

```bash
python api_server.py
```

Server runs at http://localhost:8000

服务运行在 http://localhost:8000

### Web Interface / 网页界面

Open http://localhost:8000 in your browser to access the web UI.

在浏览器中打开 http://localhost:8000 即可使用网页界面。

---

## API Reference / API 文档

### TTS with Preset Speaker / 预设说话人合成

```bash
# GET request
curl "http://localhost:8000/tts?text=Hello&speaker=aiden&language=English" -o output.wav

# With emotion instruction / 带情感指令
curl "http://localhost:8000/tts?text=你好&speaker=vivian&language=Chinese&instruct=用开心的语气说" -o output.wav
```

**Parameters / 参数:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| text | Text to synthesize / 要合成的文本 | required |
| speaker | Speaker name / 说话人 | vivian |
| language | Language / 语言 | Chinese |
| instruct | Emotion instruction / 情感指令 | optional |

### Voice Cloning / 语音克隆

```bash
curl -X POST "http://localhost:8000/clone" \
  -F "audio=@reference.wav" \
  -F "text=Hello world" \
  -F "language=English" \
  -F "ref_text=optional transcript of reference audio" \
  -o output.wav
```

**Parameters / 参数:**
| Parameter | Description |
|-----------|-------------|
| audio | Reference audio file (3-10s) / 参考音频文件 (3-10秒) |
| text | Text to synthesize / 要合成的文本 |
| language | Language / 语言 |
| ref_text | Transcript of reference audio (optional, improves quality) / 参考音频文字内容（可选，提高质量） |

### Voice Library / 声音库管理

```bash
# List saved voices / 获取已保存的声音
curl http://localhost:8000/voices

# Save a voice / 保存声音
curl -X POST "http://localhost:8000/voices/save" \
  -F "name=MyVoice" \
  -F "language=Chinese" \
  -F "audio=@reference.wav"

# Use saved voice / 使用已保存的声音
curl -X POST "http://localhost:8000/voices/{voice_id}/tts" \
  -F "text=你好世界" \
  -o output.wav

# Delete voice / 删除声音
curl -X DELETE "http://localhost:8000/voices/{voice_id}"
```

### Other Endpoints / 其他接口

```bash
# Get available speakers / 获取可用说话人
curl http://localhost:8000/speakers

# Get supported languages / 获取支持的语言
curl http://localhost:8000/languages
```

---

## Project Structure / 项目结构

```
├── api_server.py      # FastAPI server / 服务端
├── index.html         # Web UI / 网页界面
├── test_qwen_tts.py   # Test script / 测试脚本
├── models/            # Model files (not in repo) / 模型文件（不在仓库中）
└── saved_voices/      # Saved cloned voices / 保存的克隆声音
```

---

## License / 许可证

This project uses the Qwen3-TTS model. Please refer to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for model license terms.

本项目使用 Qwen3-TTS 模型，模型许可证请参考 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。
