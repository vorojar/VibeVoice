# Qwen3-TTS 语音合成服务

[English](README_EN.md)

基于阿里巴巴 Qwen3-TTS 模型的语音合成 Web 应用，支持预设说话人和语音克隆功能。

![Demo](demo.png)

> **注意**: 由于官方 SDK 暂未开放流式输出接口，本项目暂不支持流式语音输出。

---

## 功能特性

- **预设说话人**: 9 种内置声音，支持情感控制
- **语音克隆**: 3秒以上参考音频即可克隆声音
- **声音库**: 保存和管理克隆的声音，方便重复使用
- **多语言**: 支持10种语言
- **网页界面**: 支持浏览器录音的网页界面
- **REST API**: 便于集成到您的应用

## 增强功能

1. **分句进度显示** - 长文本自动分句，逐句生成并实时显示进度，可视化查看当前生成到哪一句
2. **停止生成** - 随时取消正在进行的生成，系统会在下一句边界停止
3. **字幕生成** - 自动生成与音频时间同步的 SRT/VTT 字幕文件，适合视频制作
4. **语言自动检测** - 根据输入文本自动识别语言（中/英/日/韩），并自动设置语言选择器
5. **Voice Prompt 缓存** - 声音库条目的 voice prompt 缓存到磁盘，免去重复提取，大幅加速后续生成
6. **MP3 导出** - 支持在浏览器中直接转换并下载 MP3 格式（除 WAV 外）
7. **声音设计** - 通过自然语言描述创建自定义声音（如"低沉温暖的中年男声"），多句生成自动保持音色一致
8. **句子编辑器** - 生成后可逐句选中、编辑文本、重新生成、删除或插入新句子，插入时有完整的进度反馈

## 支持的语言

中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语

## 预设说话人

| 说话人 | 语言 | 性别 |
|--------|------|------|
| vivian | 中文 | 女 |
| uncle_fu | 中文 | 男 |
| aiden | 英语 | 男 |
| serena | 英语 | 女 |
| ono_anna | 日语 | 女 |
| sohee | 韩语 | 女 |
| dylan | - | 男 |
| eric | - | 男 |
| ryan | - | 男 |

---

## 安装

### 环境要求

- Python 3.8+
- 支持 CUDA 的 GPU（建议8GB以上显存），或 macOS Apple Silicon (M1/M2/M3)
- 支持 CUDA 或 MPS 的 PyTorch

> **macOS 用户注意**: Apple Silicon 芯片通过 MPS 后端支持，会自动检测并使用 float16 精度。实际兼容性取决于 qwen-tts 库对 MPS 的支持情况。

### 安装依赖

```bash
pip install -U qwen-tts fastapi uvicorn python-multipart soundfile numpy torch
```

### 下载模型

```bash
pip install -U modelscope

# CustomVoice 模型（预设说话人）
modelscope download --model Qwen/Qwen3-TTS-1.7B-CustomVoice --local_dir ./models/Qwen3-TTS-1.7B-CustomVoice

# VoiceDesign 模型（声音设计）
modelscope download --model Qwen/Qwen3-TTS-1.7B-VoiceDesign --local_dir ./models/Qwen3-TTS-1.7B-VoiceDesign

# Base 模型（语音克隆）
modelscope download --model Qwen/Qwen3-TTS-0.6B --local_dir ./models/Qwen3-TTS-0.6B
```

---

## 使用方法

### 启动服务

```bash
python api_server.py
```

服务运行在 http://localhost:8000

### 网页界面

在浏览器中打开 http://localhost:8000 即可使用网页界面。

---

## 性能优化

### Flash Attention（推荐）

安装 Flash Attention 可提升推理速度约 **50%**。

**Linux:**
```bash
pip install flash-attn --no-build-isolation
```

**Windows:**

Windows 不支持源码编译，需使用预编译包。从 [kingbri1/flash-attention](https://github.com/kingbri1/flash-attention/releases) 下载对应版本的 wheel 文件。

示例（Python 3.10 + PyTorch 2.9 + CUDA 12.8）：
```bash
# 先升级 PyTorch
pip install torch==2.9.0 torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装预编译的 flash-attn
pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.9.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

验证安装：启动服务后不再显示 `Warning: flash-attn is not installed`

### 进一步优化

如需达到更高性能（如官方宣称的 97ms/字），需要：

| 方案 | 预期提升 | 说明 |
|------|----------|------|
| 更强 GPU | 2-5x | A100/H100 vs 消费级显卡 |
| vLLM 部署 | 2-3x | PagedAttention + 连续批处理 |
| TensorRT-LLM | 2-5x | NVIDIA 官方推理优化 |
| FP8 量化 | 1.5-2x | 需 H100 支持 |

> 消费级显卡（RTX 40系）+ Flash Attention 达到 ~1.4s/字 是合理水平。

---

## API 文档

### 预设说话人合成

```bash
# GET 请求
curl "http://localhost:8000/tts?text=你好&speaker=vivian&language=Chinese" -o output.wav

# 带情感指令
curl "http://localhost:8000/tts?text=你好&speaker=vivian&language=Chinese&instruct=用开心的语气说" -o output.wav
```

**参数:**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| text | 要合成的文本 | 必填 |
| speaker | 说话人 | vivian |
| language | 语言 | Chinese |
| instruct | 情感指令 | 可选 |

### 语音克隆

```bash
curl -X POST "http://localhost:8000/clone" \
  -F "audio=@reference.wav" \
  -F "text=你好世界" \
  -F "language=Chinese" \
  -F "ref_text=参考音频的文字内容" \
  -o output.wav
```

**参数:**
| 参数 | 说明 |
|------|------|
| audio | 参考音频文件（3-10秒） |
| text | 要合成的文本 |
| language | 语言 |
| ref_text | 参考音频文字内容（可选，提高质量） |

### 声音库管理

```bash
# 获取已保存的声音
curl http://localhost:8000/voices

# 保存声音
curl -X POST "http://localhost:8000/voices/save" \
  -F "name=我的声音" \
  -F "language=Chinese" \
  -F "audio=@reference.wav"

# 使用已保存的声音
curl -X POST "http://localhost:8000/voices/{voice_id}/tts" \
  -F "text=你好世界" \
  -o output.wav

# 删除声音
curl -X DELETE "http://localhost:8000/voices/{voice_id}"
```

### 其他接口

```bash
# 获取可用说话人
curl http://localhost:8000/speakers

# 获取支持的语言
curl http://localhost:8000/languages
```

---

## 项目结构

```
├── api_server.py      # FastAPI 服务端
├── index.html         # 网页界面
├── test_qwen_tts.py   # 测试脚本
├── models/            # 模型文件（不在仓库中）
└── saved_voices/      # 保存的克隆声音
```

---

## 更新日志

### v0.3.0 (2025-02-18)

**波形可视化播放器**
- 集成 WaveSurfer.js 替代原来的简易进度条，显示音频波形
- 播放时字幕区域高亮当前句子，已播过的句子变淡

**句子编辑器**
- 生成完成后进入句子编辑视图，支持逐句操作
- 单击选中句子，双击编辑文本
- 单句重新生成（带 spinner 反馈），支持撤销（↩）回退到上一版本
- 单句删除（带确认）
- 句间添加新句子（带占位行、spinner、禁用其他按钮，完成后自动跳播）
- 句间停顿控制（0x ~ 2x 滑块，实时调整句间静音时长）
- 逐句试听（▶ 按钮，播放单句音频片段）

**会话持久化 (IndexedDB)**
- 生成结果（分句音频、文本、字幕、参数）自动保存到 IndexedDB
- 刷新页面后自动恢复上次会话，无需重新生成
- 保存句间停顿倍率等编辑状态

**声音设计跨句音色一致性**
- 多句生成时，第一句用 design 模型确立音色，后续句子自动切换为 clone 模型 + 首句 prompt，确保全部句子音色一致
- 单句重新生成和添加句子同样复用缓存的 voice prompt，不会跑偏音色
- 单句文本无额外开销，仍走纯 design 模型
- clone 模型未加载时自动 fallback 到 design 模型

**后端**
- 所有 4 个进度端点（tts/clone/design/saved_voice）返回逐句 base64 音频数组 `sentence_audios`
- 新增 `POST /regenerate` 端点，支持单句重新生成（preset/clone/design/saved_voice 四种模式）
- 克隆会话 prompt 缓存机制（`clone_session_prompts`），1 小时过期自动清理

### v0.2.0

- 分句进度显示、停止生成、字幕生成
- Voice Prompt 磁盘缓存、MP3 导出
- 声音设计模式、语言自动检测

### v0.1.0

- 预设说话人合成（9 种声音 + 情感控制）
- 语音克隆（录音/上传参考音频）
- 声音库管理
- 多语言支持（10 种语言）
- REST API、中英双语界面

---

## 许可证

本项目使用 Qwen3-TTS 模型，模型许可证请参考 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)。
