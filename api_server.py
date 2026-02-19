import io
import os
import json
import tempfile
import time
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil

# 延迟导入：torch 和 qwen_tts 启动开销大，首次使用时才加载
torch = None
Qwen3TTSModel = None

def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch

def _ensure_qwen_tts():
    global Qwen3TTSModel
    _ensure_torch()
    if Qwen3TTSModel is None:
        from qwen_tts import Qwen3TTSModel as _cls
        Qwen3TTSModel = _cls


def get_generation_stats(text: str, start_time: float, mode: str = "TTS") -> dict:
    """计算生成统计信息"""
    elapsed = time.time() - start_time
    char_count = len(text)
    avg_per_char = elapsed / char_count if char_count > 0 else 0
    print(f"[{mode}] 生成完成: {char_count} 字 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")
    return {
        "char_count": char_count,
        "elapsed": round(elapsed, 2),
        "avg_per_char": round(avg_per_char, 3),
    }

app = FastAPI(title="VibeVoice API", version="1.0.0")

# 添加 CORS 支持，允许 Tauri 应用访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境可以限制）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
VOICES_DIR = BASE_DIR / "saved_voices"
VOICES_DIR.mkdir(exist_ok=True)

# 全局模型
model_custom = None  # CustomVoice 模型
model_clone = None   # Base 模型（用于语音克隆）
model_design = None  # VoiceDesign 模型
SAMPLE_RATE = 24000

# 模型加载状态: unloaded / loading / loaded
model_status = {
    "custom": "unloaded",
    "design": "unloaded",
    "clone": "unloaded",
}

# 模型配置
MODEL_CONFIGS = {
    "custom": {
        "path": "./models/Qwen3-TTS-1.7B-CustomVoice",
        "name": "CustomVoice 1.7B",
    },
    "design": {
        "path": "./models/Qwen3-TTS-1.7B-VoiceDesign",
        "name": "VoiceDesign 1.7B",
    },
    "clone": {
        "path": "./models/Qwen3-TTS-0.6B",
        "name": "Base 0.6B",
    },
}

# 已保存的克隆声音 prompt 缓存 (voice_id -> VoiceClonePromptItem)
saved_voice_prompts = {}

# 克隆会话 prompt 缓存 (session_id -> {"prompt": VoiceClonePromptItem, "created_at": float})
clone_session_prompts = {}


def save_voice_prompt(voice_id: str, prompt_item):
    """将 VoiceClonePromptItem 保存到磁盘"""
    _ensure_torch()
    voice_dir = VOICES_DIR / voice_id
    prompt_path = voice_dir / "prompt.pt"

    # 保存为 dict 格式
    prompt_data = {
        "ref_code": prompt_item.ref_code,
        "ref_spk_embedding": prompt_item.ref_spk_embedding,
        "x_vector_only_mode": prompt_item.x_vector_only_mode,
        "icl_mode": prompt_item.icl_mode,
        "ref_text": prompt_item.ref_text,
    }
    torch.save(prompt_data, prompt_path)
    print(f"[缓存] 已保存 voice prompt: {voice_id}")


def load_voice_prompt(voice_id: str):
    """从磁盘加载 VoiceClonePromptItem"""
    _ensure_torch()
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    voice_dir = VOICES_DIR / voice_id
    prompt_path = voice_dir / "prompt.pt"

    if not prompt_path.exists():
        return None

    try:
        prompt_data = torch.load(prompt_path, weights_only=False)
        prompt_item = VoiceClonePromptItem(
            ref_code=prompt_data["ref_code"],
            ref_spk_embedding=prompt_data["ref_spk_embedding"],
            x_vector_only_mode=prompt_data["x_vector_only_mode"],
            icl_mode=prompt_data["icl_mode"],
            ref_text=prompt_data.get("ref_text"),
        )
        print(f"[缓存] 已加载 voice prompt: {voice_id}")
        return prompt_item
    except Exception as e:
        print(f"[缓存] 加载失败 {voice_id}: {e}")
        return None


def get_or_create_voice_prompt(voice_id: str, audio_path: str, ref_text: str = ""):
    """获取或创建 voice prompt（带缓存）"""
    # 1. 先检查内存缓存
    if voice_id in saved_voice_prompts:
        print(f"[缓存] 命中内存: {voice_id}")
        return saved_voice_prompts[voice_id]

    # 2. 检查磁盘缓存
    prompt_item = load_voice_prompt(voice_id)
    if prompt_item:
        saved_voice_prompts[voice_id] = prompt_item
        return prompt_item

    # 3. 创建新的 prompt
    if model_clone is None:
        raise RuntimeError("Clone model not loaded")

    print(f"[缓存] 创建新 prompt: {voice_id}")
    use_x_vector_only = not ref_text
    prompts = model_clone.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=ref_text if ref_text else None,
        x_vector_only_mode=use_x_vector_only,
    )
    prompt_item = prompts[0]

    # 保存到磁盘和内存
    save_voice_prompt(voice_id, prompt_item)
    saved_voice_prompts[voice_id] = prompt_item

    return prompt_item

# 支持的说话人
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

# 支持的语言
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


def get_device_config():
    """自动检测最佳设备和数据类型"""
    _ensure_torch()
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16  # MPS 对 bfloat16 支持有限
    else:
        return "cpu", torch.float32


def load_model_sync(model_type: str):
    """同步加载指定模型"""
    global model_custom, model_clone, model_design, model_status

    if model_status[model_type] != "unloaded":
        return

    model_status[model_type] = "loading"
    config = MODEL_CONFIGS[model_type]

    try:
        _ensure_qwen_tts()
        device_map, dtype = get_device_config()
        print(f"正在加载 {config['name']} 模型... (设备: {device_map}, 精度: {dtype})")

        model = Qwen3TTSModel.from_pretrained(
            config["path"],
            device_map=device_map,
            dtype=dtype,
        )

        if model_type == "custom":
            model_custom = model
        elif model_type == "design":
            model_design = model
        elif model_type == "clone":
            model_clone = model

        model_status[model_type] = "loaded"
        print(f"{config['name']} 模型加载完成！")
    except Exception as e:
        model_status[model_type] = "unloaded"
        print(f"{config['name']} 模型加载失败: {e}")
        raise


@app.get("/model/status")
async def get_model_status():
    """获取所有模型的加载状态"""
    return {"status": model_status}


@app.post("/model/load/{model_type}")
async def load_model_endpoint(model_type: str):
    """触发加载指定模型"""
    if model_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"未知的模型类型: {model_type}")

    if model_status[model_type] == "loaded":
        return {"status": "loaded", "message": "模型已加载"}

    if model_status[model_type] == "loading":
        return {"status": "loading", "message": "模型正在加载中"}

    # 在后台线程中加载模型
    import threading
    thread = threading.Thread(target=load_model_sync, args=(model_type,))
    thread.start()

    return {"status": "loading", "message": f"开始加载 {MODEL_CONFIGS[model_type]['name']}"}


@app.get("/")
async def root():
    """返回演示页面"""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api")
async def api_info():
    return {
        "service": "VibeVoice API",
        "version": "1.0.0",
        "endpoints": {
            "/tts": "文字转语音 - 预设说话人 (GET/POST)",
            "/clone": "语音克隆 - 上传参考音频 (POST)",
            "/speakers": "获取可用说话人列表",
            "/languages": "获取支持的语言列表",
        }
    }


@app.get("/speakers")
async def get_speakers():
    """获取可用的说话人列表"""
    return {"speakers": SPEAKERS}


@app.get("/languages")
async def get_languages():
    """获取支持的语言列表"""
    return {"languages": LANGUAGES}


@app.get("/tts")
async def tts(
    text: str = Query(..., description="要合成的文本"),
    speaker: str = Query("vivian", description="说话人"),
    language: str = Query("Chinese", description="语言"),
    instruct: str = Query(None, description="情感指令，如：用开心的语气说"),
):
    """
    文字转语音 API（预设说话人）

    参数:
    - text: 要合成的文本
    - speaker: 说话人
    - language: 语言
    - instruct: 情感指令（可选）

    返回:
    - WAV 音频文件
    """
    # 检查模型是否加载
    if model_status["custom"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "custom", "status": model_status["custom"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if speaker not in SPEAKERS:
        raise HTTPException(status_code=400, detail=f"不支持的说话人: {speaker}。支持: {SPEAKERS}")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}。支持: {LANGUAGES}")

    try:
        print(f"[TTS] 开始生成: {len(text)} 字 | 说话人: {speaker} | 语言: {language}")
        start_time = time.time()

        wavs, sr = model_custom.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        stats = get_generation_stats(text, start_time, "TTS")

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.wav",
                "X-Char-Count": str(stats["char_count"]),
                "X-Elapsed": str(stats["elapsed"]),
                "X-Avg-Per-Char": str(stats["avg_per_char"]),
                "Access-Control-Expose-Headers": "X-Char-Count, X-Elapsed, X-Avg-Per-Char",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def tts_post(
    text: str = Query(..., description="要合成的文本"),
    speaker: str = Query("vivian", description="说话人"),
    language: str = Query("Chinese", description="语言"),
    instruct: str = Query(None, description="情感指令"),
):
    """POST 方式的 TTS API"""
    return await tts(text=text, speaker=speaker, language=language, instruct=instruct)


import re

def split_text_to_sentences(text: str, min_length: int = 10) -> list:
    """
    智能分句：按标点分割，合并过短的句子

    - 支持连续标点 ???、!!!、。。。
    - 短句自动合并，直到达到 min_length
    """
    # 匹配一个或多个连续的中英文标点（含闭合引号），或冒号引出引语，或换行
    pattern = r'([。！？；.!?;][。！？；.!?;"\u201C\u201D\u300C\u300D\'\u2018\u2019]*|[：:](?=["\u201C\u201D\u300C\u300D\'\u2018\u2019])|\n)'
    parts = re.split(pattern, text)

    # split 带捕获组：奇数索引是分隔符，用索引判定（避免 lookahead 在隔离部分失效）
    raw_sentences = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            if current.strip():
                raw_sentences.append(current.strip())
            current = ""
    if current.strip():
        raw_sentences.append(current.strip())

    # 如果没有分割出句子，返回整个文本
    if not raw_sentences:
        return [text] if text.strip() else []

    # 合并过短的句子（以冒号结尾的句子强制独立，用于旁白/引语分离）
    merged = []
    buffer = ""
    for sentence in raw_sentences:
        buffer += sentence
        if len(buffer) >= min_length or buffer.endswith('：') or buffer.endswith(':'):
            merged.append(buffer)
            buffer = ""

    # 处理剩余内容
    if buffer:
        if merged:
            # 追加到最后一句
            merged[-1] += buffer
        else:
            merged.append(buffer)

    return merged


def split_text_to_paragraphs(text: str, max_chars: int = 300) -> list:
    """
    按段落分割文本，超长段落按句号再拆

    - 按换行符拆自然段落
    - 超过 max_chars 的段落按句号分割成子段
    - 空段落跳过
    """
    raw_paragraphs = text.split('\n')
    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            paragraphs.append(para)
        else:
            # 超长段落按句号分割
            sub_parts = re.split(r'(?<=[。！？.!?])', para)
            current = ""
            for part in sub_parts:
                if not part:
                    continue
                if len(current) + len(part) <= max_chars:
                    current += part
                else:
                    if current:
                        paragraphs.append(current)
                    current = part
            if current:
                paragraphs.append(current)
    # 如果没分出段落（例如无换行的短文本），返回整个文本
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    return paragraphs



@app.get("/tts/stream")
async def tts_stream(
    text: str = Query(..., description="要合成的文本"),
    speaker: str = Query("vivian", description="说话人"),
    language: str = Query("Chinese", description="语言"),
    instruct: str = Query(None, description="情感指令"),
):
    """
    流式文字转语音 API（SSE）- 按句子分割生成

    返回:
    - SSE 事件流，每个事件包含一句话的音频
    - 事件格式: data: {"audio": "base64...", "sample_rate": 24000, "chunk_index": 0, "text": "句子内容"}
    - 结束事件: data: {"done": true, "total_chunks": N}
    """
    import base64

    if model_status["custom"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "custom", "status": model_status["custom"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if speaker not in SPEAKERS:
        raise HTTPException(status_code=400, detail=f"不支持的说话人: {speaker}")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}")

    def generate_audio_stream():
        try:
            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'mode': 'token_streaming'})}\n\n"

            chunk_index = 0
            for audio_chunk, sr in model_custom.generate_custom_voice_streaming(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                chunk_size=6,  # 6 tokens ≈ 0.5秒音频，首包更快
            ):
                # 将音频编码为 base64
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, audio_chunk, sr, format='WAV')
                audio_buffer.seek(0)
                audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

                yield f"data: {json.dumps({'audio': audio_base64, 'sample_rate': sr, 'chunk_index': chunk_index})}\n\n"
                chunk_index += 1

            # 发送结束信号
            yield f"data: {json.dumps({'done': True, 'total_chunks': chunk_index})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        generate_audio_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/tts/progress")
async def tts_with_progress(
    request: Request,
    text: str = Query(..., description="要合成的文本"),
    speaker: str = Query("vivian", description="说话人"),
    language: str = Query("Chinese", description="语言"),
    instruct: str = Query(None, description="情感指令"),
):
    """
    分句生成 TTS（带进度）

    - SSE 推送进度: {"progress": {"current": 1, "total": 10}}
    - 最后推送完整音频: {"done": true, "audio": "base64..."}
    - 支持客户端断开时停止生成
    """
    import base64
    import asyncio

    if model_status["custom"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "custom", "status": model_status["custom"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if speaker not in SPEAKERS:
        raise HTTPException(status_code=400, detail=f"不支持的说话人: {speaker}")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}")

    async def generate_with_progress():
        completed = 0
        total = 0
        try:
            # 分段落，再分句
            paragraphs = split_text_to_paragraphs(text, max_chars=300)
            all_sentences = []
            para_sent_map = []  # [(start_idx, end_idx), ...]
            for para in paragraphs:
                start = len(all_sentences)
                sents = split_text_to_sentences(para, min_length=10)
                all_sentences.extend(sents)
                para_sent_map.append((start, len(all_sentences)))

            total = len(all_sentences)
            total_chars = len(text)
            num_paragraphs = len(paragraphs)

            print(f"[TTS进度] 开始生成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 说话人: {speaker}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars, 'paragraphs': num_paragraphs})}\n\n"

            # 收集所有音频和字幕时间
            all_audio = []
            sentence_audios_b64 = []
            subtitles = []
            current_time = 0.0
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for p_idx, paragraph in enumerate(paragraphs):
                sent_start, sent_end = para_sent_map[p_idx]
                para_sentences = all_sentences[sent_start:sent_end]

                # 逐句生成
                for c_idx, sentence in enumerate(para_sentences):
                    if await request.is_disconnected():
                        print(f"[TTS进度] 客户端已断开，停止生成 (句 {sent_start+c_idx}/{total})")
                        return

                    wavs, sr = await loop.run_in_executor(
                        None,
                        lambda s=sentence: model_custom.generate_custom_voice(
                            text=s,
                            language=language,
                            speaker=speaker,
                            instruct=instruct,
                        )
                    )

                    sample_rate = sr
                    chunk = wavs[0]
                    all_audio.append(chunk)

                    sent_buf = io.BytesIO()
                    sf.write(sent_buf, chunk, sr, format='WAV')
                    sent_buf.seek(0)
                    sentence_audios_b64.append(base64.b64encode(sent_buf.read()).decode('utf-8'))

                    duration = len(chunk) / sr
                    subtitles.append({"text": sentence, "start": round(current_time, 3), "end": round(current_time + duration, 3)})
                    current_time += duration

                    completed = sent_start + c_idx + 1
                    progress_data = {
                        "progress": {
                            "current": completed,
                            "total": total,
                            "percent": round(completed / total * 100),
                            "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                            "paragraph": p_idx + 1,
                            "total_paragraphs": num_paragraphs,
                        }
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

                print(f"[TTS进度] 段落 {p_idx+1}/{num_paragraphs} 完成 (句 {sent_start+1}-{sent_end}/{total}): {paragraph[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[TTS进度] 全部完成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

            # 编码为 base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, merged_audio, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            # 发送完成信号和音频
            done_data = {
                "done": True,
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "subtitles": subtitles,
                "sentence_audios": sentence_audios_b64,
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            print(f"[TTS进度] 生成被取消 ({completed}/{total})")
        except GeneratorExit:
            print(f"[TTS进度] 客户端断开连接 ({completed}/{total})")
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/clone")
async def voice_clone(
    text: str = Form(..., description="要合成的文本"),
    language: str = Form("Chinese", description="语言"),
    ref_text: str = Form("", description="参考音频的文字内容（可选，提供可提高质量）"),
    audio: UploadFile = File(..., description="参考音频文件（WAV/MP3）"),
):
    """
    语音克隆 API

    参数:
    - text: 要合成的文本
    - language: 语言
    - ref_text: 参考音频的文字内容（可选）
    - audio: 参考音频文件

    返回:
    - WAV 音频文件（使用克隆的声音）
    """
    # 检查模型是否加载
    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "clone", "status": model_status["clone"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}。支持: {LANGUAGES}")

    # 保存上传的音频到临时文件
    temp_file = None
    try:
        # 读取上传的音频
        audio_content = await audio.read()

        # 保存到临时文件
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(audio_content)
        temp_file.close()

        # 生成语音
        # 如果没有提供参考文本，使用 x_vector_only_mode
        print(f"[克隆] 开始生成: {len(text)} 字 | 语言: {language}")
        start_time = time.time()

        use_x_vector_only = not ref_text
        wavs, sr = model_clone.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=temp_file.name,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=use_x_vector_only,
        )

        stats = get_generation_stats(text, start_time, "克隆")

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=clone_output.wav",
                "X-Char-Count": str(stats["char_count"]),
                "X-Elapsed": str(stats["elapsed"]),
                "X-Avg-Per-Char": str(stats["avg_per_char"]),
                "Access-Control-Expose-Headers": "X-Char-Count, X-Elapsed, X-Avg-Per-Char",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.post("/clone/progress")
async def voice_clone_progress(
    request: Request,
    text: str = Form(..., description="要合成的文本"),
    language: str = Form("Chinese", description="语言"),
    ref_text: str = Form("", description="参考音频的文字内容"),
    audio: UploadFile = File(..., description="参考音频文件"),
):
    """语音克隆（带进度）"""
    import base64
    import asyncio

    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "clone", "status": model_status["clone"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}")

    # 保存上传的音频到临时文件
    audio_content = await audio.read()
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_content)
    temp_file.close()
    temp_path = temp_file.name

    async def generate_with_progress():
        completed = 0
        total = 0
        try:
            # 分段落，再分句
            paragraphs = split_text_to_paragraphs(text, max_chars=300)
            all_sentences = []
            para_sent_map = []
            for para in paragraphs:
                start = len(all_sentences)
                sents = split_text_to_sentences(para, min_length=10)
                all_sentences.extend(sents)
                para_sent_map.append((start, len(all_sentences)))

            total = len(all_sentences)
            total_chars = len(text)
            num_paragraphs = len(paragraphs)

            print(f"[克隆进度] 开始生成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 语言: {language}")
            start_time = time.time()

            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars, 'paragraphs': num_paragraphs})}\n\n"

            # 生成 voice prompt
            use_x_vector_only = not ref_text
            prompts = model_clone.create_voice_clone_prompt(
                ref_audio=temp_path,
                ref_text=ref_text if ref_text else None,
                x_vector_only_mode=use_x_vector_only,
            )
            voice_prompt = prompts[0]

            # 缓存 clone session prompt
            import uuid as _uuid
            clone_session_id = _uuid.uuid4().hex
            clone_session_prompts[clone_session_id] = {
                "prompt": voice_prompt,
                "created_at": time.time(),
            }
            # 清理过期缓存（1小时）
            expired = [k for k, v in clone_session_prompts.items() if time.time() - v["created_at"] > 3600]
            for k in expired:
                del clone_session_prompts[k]

            all_audio = []
            sentence_audios_b64 = []
            subtitles = []
            current_time = 0.0
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for p_idx, paragraph in enumerate(paragraphs):
                sent_start, sent_end = para_sent_map[p_idx]
                para_sentences = all_sentences[sent_start:sent_end]

                # 逐句生成
                for c_idx, sentence in enumerate(para_sentences):
                    if await request.is_disconnected():
                        print(f"[克隆进度] 客户端已断开，停止生成 (句 {sent_start+c_idx}/{total})")
                        return

                    wavs, sr = await loop.run_in_executor(
                        None,
                        lambda s=sentence, vp=voice_prompt: model_clone.generate_voice_clone(
                            text=s,
                            language=language,
                            voice_clone_prompt=[vp],
                        )
                    )

                    sample_rate = sr
                    chunk = wavs[0]
                    all_audio.append(chunk)

                    sent_buf = io.BytesIO()
                    sf.write(sent_buf, chunk, sr, format='WAV')
                    sent_buf.seek(0)
                    sentence_audios_b64.append(base64.b64encode(sent_buf.read()).decode('utf-8'))

                    duration = len(chunk) / sr
                    subtitles.append({"text": sentence, "start": round(current_time, 3), "end": round(current_time + duration, 3)})
                    current_time += duration

                    completed = sent_start + c_idx + 1
                    progress_data = {
                        "progress": {
                            "current": completed,
                            "total": total,
                            "percent": round(completed / total * 100),
                            "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                            "paragraph": p_idx + 1,
                            "total_paragraphs": num_paragraphs,
                        }
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

                print(f"[克隆进度] 段落 {p_idx+1}/{num_paragraphs} 完成 (句 {sent_start+1}-{sent_end}/{total}): {paragraph[:30]}...")

            merged_audio = np.concatenate(all_audio)

            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[克隆进度] 全部完成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, merged_audio, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            done_data = {
                "done": True,
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "subtitles": subtitles,
                "sentence_audios": sentence_audios_b64,
                "clone_prompt_id": clone_session_id,
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            print(f"[克隆进度] 生成被取消 ({completed}/{total})")
        except GeneratorExit:
            print(f"[克隆进度] 客户端断开连接 ({completed}/{total})")
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/design")
async def voice_design(
    text: str = Form(..., description="要合成的文本"),
    language: str = Form("Chinese", description="语言"),
    instruct: str = Form(..., description="声音描述，如：低沉沙哑的中年男声"),
):
    """
    声音设计 API

    通过自然语言描述生成特定风格的声音

    参数:
    - text: 要合成的文本
    - language: 语言
    - instruct: 声音描述（必填）

    返回:
    - WAV 音频文件
    """
    # 检查模型是否加载
    if model_status["design"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "design", "status": model_status["design"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if not instruct:
        raise HTTPException(status_code=400, detail="instruct 声音描述不能为空")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}。支持: {LANGUAGES}")

    try:
        print(f"[设计] 开始生成: {len(text)} 字 | 语言: {language}")
        start_time = time.time()

        wavs, sr = model_design.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )

        stats = get_generation_stats(text, start_time, "设计")

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=design_output.wav",
                "X-Char-Count": str(stats["char_count"]),
                "X-Elapsed": str(stats["elapsed"]),
                "X-Avg-Per-Char": str(stats["avg_per_char"]),
                "Access-Control-Expose-Headers": "X-Char-Count, X-Elapsed, X-Avg-Per-Char",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/design/progress")
async def voice_design_progress(
    request: Request,
    text: str = Query(..., description="要合成的文本"),
    language: str = Query("Chinese", description="语言"),
    instruct: str = Query(..., description="声音描述"),
):
    """
    声音设计 API（带进度）

    - SSE 推送进度: {"progress": {"current": 1, "total": 10, "percent": 10}}
    - 最后推送完整音频: {"done": true, "audio": "base64..."}
    - 支持客户端断开时停止生成
    """
    import base64
    import asyncio

    if model_status["design"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "design", "status": model_status["design"]})

    if not text:
        raise HTTPException(status_code=400, detail="text 参数不能为空")

    if not instruct:
        raise HTTPException(status_code=400, detail="instruct 声音描述不能为空")

    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"不支持的语言: {language}")

    async def generate_with_progress():
        completed = 0
        total = 0
        try:
            # 分段落，再分句
            paragraphs = split_text_to_paragraphs(text, max_chars=300)
            all_sentences = []
            para_sent_map = []
            for para in paragraphs:
                start = len(all_sentences)
                sents = split_text_to_sentences(para, min_length=10)
                all_sentences.extend(sents)
                para_sent_map.append((start, len(all_sentences)))

            total = len(all_sentences)
            total_chars = len(text)
            num_paragraphs = len(paragraphs)

            # 多句时需要 clone 模型来保持音色一致
            use_clone_for_consistency = total > 1 and model_status["clone"] == "loaded"
            if total > 1 and model_status["clone"] != "loaded":
                print(f"[设计进度] 警告: clone 模型未加载，多句生成将使用 design 模型（音色可能不一致）")

            print(f"[设计进度] 开始生成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 语言: {language} | 跨段一致: {use_clone_for_consistency}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars, 'paragraphs': num_paragraphs})}\n\n"

            # 收集所有音频和字幕时间
            all_audio = []
            sentence_audios_b64 = []
            subtitles = []
            current_time = 0.0
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()
            voice_prompt = None
            clone_session_id = None

            global_sent_idx = 0  # 全局句子计数，用于判断是否第一句

            for p_idx, paragraph in enumerate(paragraphs):
                sent_start, sent_end = para_sent_map[p_idx]
                para_sentences = all_sentences[sent_start:sent_end]

                # 逐句生成
                for c_idx, sentence in enumerate(para_sentences):
                    if await request.is_disconnected():
                        print(f"[设计进度] 客户端已断开，停止生成 (句 {sent_start+c_idx}/{total})")
                        return

                    if global_sent_idx == 0 or not use_clone_for_consistency:
                        # 第一句（或不使用跨段一致时所有句子）：用 design 模型
                        wavs, sr = await loop.run_in_executor(
                            None,
                            lambda s=sentence: model_design.generate_voice_design(
                                text=s,
                                language=language,
                                instruct=instruct,
                            )
                        )

                        # 第一句且多句模式：用生成的音频创建 clone prompt
                        if global_sent_idx == 0 and use_clone_for_consistency:
                            import uuid as _uuid
                            voice_prompt = await loop.run_in_executor(
                                None,
                                lambda audio=wavs[0], _sr=sr, ref=sentence: model_clone.create_voice_clone_prompt(
                                    ref_audio=(audio, _sr),
                                    ref_text=ref,
                                )
                            )
                            voice_prompt = voice_prompt[0]
                            clone_session_id = _uuid.uuid4().hex
                            clone_session_prompts[clone_session_id] = {
                                "prompt": voice_prompt,
                                "created_at": time.time(),
                            }
                            expired = [k for k, v in clone_session_prompts.items() if time.time() - v["created_at"] > 3600]
                            for k in expired:
                                del clone_session_prompts[k]
                            print(f"[设计进度] 已从第一句创建 clone prompt，后续句子将使用 clone 模型")
                    else:
                        # 后续句：用 clone 模型 + voice_prompt（保持音色一致）
                        wavs, sr = await loop.run_in_executor(
                            None,
                            lambda s=sentence, vp=voice_prompt: model_clone.generate_voice_clone(
                                text=s,
                                language=language,
                                voice_clone_prompt=[vp],
                            )
                        )

                    sample_rate = sr
                    chunk = wavs[0]
                    all_audio.append(chunk)

                    sent_buf = io.BytesIO()
                    sf.write(sent_buf, chunk, sr, format='WAV')
                    sent_buf.seek(0)
                    sentence_audios_b64.append(base64.b64encode(sent_buf.read()).decode('utf-8'))

                    duration = len(chunk) / sr
                    subtitles.append({"text": sentence, "start": round(current_time, 3), "end": round(current_time + duration, 3)})
                    current_time += duration

                    global_sent_idx += 1
                    completed = sent_start + c_idx + 1
                    progress_data = {
                        "progress": {
                            "current": completed,
                            "total": total,
                            "percent": round(completed / total * 100),
                            "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                            "paragraph": p_idx + 1,
                            "total_paragraphs": num_paragraphs,
                        }
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

                print(f"[设计进度] 段落 {p_idx+1}/{num_paragraphs} 完成 (句 {sent_start+1}-{sent_end}/{total}): {paragraph[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[设计进度] 全部完成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

            # 编码为 base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, merged_audio, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            # 发送完成信号和音频
            done_data = {
                "done": True,
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "subtitles": subtitles,
                "sentence_audios": sentence_audios_b64,
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            if clone_session_id:
                done_data["clone_prompt_id"] = clone_session_id
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            print(f"[设计进度] 生成被取消 ({completed}/{total})")
        except GeneratorExit:
            print(f"[设计进度] 客户端断开连接 ({completed}/{total})")
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/regenerate")
async def regenerate_sentence(
    sentence_text: str = Form(..., description="新的句子文本"),
    mode: str = Form(..., description="模式: preset / clone / design / saved_voice"),
    language: str = Form("Chinese", description="语言"),
    speaker: str = Form(None, description="说话人（preset 模式）"),
    instruct: str = Form(None, description="情感指令（preset 模式）或声音描述（design 模式）"),
    voice_id: str = Form(None, description="声音库 ID（saved_voice 模式）"),
    clone_prompt_id: str = Form(None, description="克隆会话 ID（clone 模式）"),
):
    """
    单句重新生成

    只生成单句音频返回，合并和字幕计算由前端完成
    """
    import base64

    try:
        print(f"[重新生成] 模式: {mode} | {sentence_text[:30]}...")
        start_time = time.time()

        if mode == "preset":
            if model_status["custom"] != "loaded":
                raise HTTPException(status_code=503, detail="CustomVoice 模型未加载")
            wavs, sr = model_custom.generate_custom_voice(
                text=sentence_text,
                language=language,
                speaker=speaker or "vivian",
                instruct=instruct,
            )
        elif mode == "clone":
            if model_status["clone"] != "loaded":
                raise HTTPException(status_code=503, detail="Clone 模型未加载")
            if not clone_prompt_id or clone_prompt_id not in clone_session_prompts:
                raise HTTPException(status_code=400, detail="克隆会话已过期，请重新生成")
            voice_prompt = clone_session_prompts[clone_prompt_id]["prompt"]
            wavs, sr = model_clone.generate_voice_clone(
                text=sentence_text,
                language=language,
                voice_clone_prompt=[voice_prompt],
            )
        elif mode == "design":
            if clone_prompt_id and clone_prompt_id in clone_session_prompts:
                # 用缓存的 clone prompt 保持音色一致
                if model_status["clone"] != "loaded":
                    raise HTTPException(status_code=503, detail="Clone 模型未加载")
                voice_prompt = clone_session_prompts[clone_prompt_id]["prompt"]
                wavs, sr = model_clone.generate_voice_clone(
                    text=sentence_text,
                    language=language,
                    voice_clone_prompt=[voice_prompt],
                )
            else:
                # fallback: 无缓存，用 design 模型（音色可能不一致）
                if model_status["design"] != "loaded":
                    raise HTTPException(status_code=503, detail="VoiceDesign 模型未加载")
                wavs, sr = model_design.generate_voice_design(
                    text=sentence_text,
                    language=language,
                    instruct=instruct or "",
                )
        elif mode == "saved_voice":
            if model_status["clone"] != "loaded":
                raise HTTPException(status_code=503, detail="Clone 模型未加载")
            if not voice_id:
                raise HTTPException(status_code=400, detail="缺少 voice_id")
            voice_dir = VOICES_DIR / voice_id
            if not voice_dir.exists():
                raise HTTPException(status_code=404, detail="声音不存在")
            meta_file = voice_dir / "meta.json"
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            audio_path = voice_dir / meta["audio_file"]
            ref_text = meta.get("ref_text", "")
            voice_prompt = get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)
            wavs, sr = model_clone.generate_voice_clone(
                text=sentence_text,
                language=language,
                voice_clone_prompt=[voice_prompt],
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

        elapsed = time.time() - start_time
        print(f"[重新生成] 完成: {elapsed:.2f}s")

        # 编码新句音频为 base64
        new_sent_buf = io.BytesIO()
        sf.write(new_sent_buf, wavs[0], sr, format='WAV')
        new_sent_buf.seek(0)
        new_audio_b64 = base64.b64encode(new_sent_buf.read()).decode('utf-8')

        return JSONResponse({
            "audio": new_audio_b64,
            "sample_rate": sr,
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def get_saved_voices():
    """获取已保存的克隆声音列表"""
    voices = []
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            meta_file = voice_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    voice_id = voice_dir.name
                    # 检查是否已缓存
                    cached = (voice_id in saved_voice_prompts) or (voice_dir / "prompt.pt").exists()
                    voices.append({
                        "id": voice_id,
                        "name": meta.get("name", voice_id),
                        "language": meta.get("language", "Chinese"),
                        "created_at": meta.get("created_at", ""),
                        "cached": cached,
                    })
    return {"voices": voices}


@app.post("/voices/{voice_id}/cache")
async def cache_voice_prompt(voice_id: str):
    """为指定声音预缓存 prompt"""
    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail="克隆模型未加载")

    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="声音不存在")

    meta_file = voice_dir / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    audio_path = voice_dir / meta["audio_file"]
    ref_text = meta.get("ref_text", "")

    try:
        get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)
        return {"success": True, "message": "缓存成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voices/cache-all")
async def cache_all_voice_prompts():
    """为所有未缓存的声音预缓存 prompt"""
    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail="克隆模型未加载")

    cached = []
    failed = []

    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            voice_id = voice_dir.name
            # 跳过已缓存的
            if (voice_dir / "prompt.pt").exists():
                continue

            meta_file = voice_dir / "meta.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                audio_path = voice_dir / meta["audio_file"]
                ref_text = meta.get("ref_text", "")
                get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)
                cached.append(voice_id)
            except Exception as e:
                failed.append({"id": voice_id, "error": str(e)})

    return {"cached": cached, "failed": failed, "message": f"已缓存 {len(cached)} 个声音"}


@app.post("/voices/design-preview")
async def design_voice_preview(
    text: str = Form(..., description="试听文本"),
    language: str = Form("Chinese", description="语言"),
    instruct: str = Form(..., description="声音描述"),
):
    """
    设计声音预览：用 design 模型生成试听音频

    返回 base64 音频，前端播放试听
    """
    import base64

    if model_status["design"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "design", "status": model_status["design"]})

    if not text:
        raise HTTPException(status_code=400, detail="试听文本不能为空")
    if not instruct:
        raise HTTPException(status_code=400, detail="声音描述不能为空")

    try:
        print(f"[设计预览] 生成试听: {len(text)} 字 | 描述: {instruct[:30]}")
        start_time = time.time()

        wavs, sr = model_design.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )

        elapsed = time.time() - start_time
        print(f"[设计预览] 完成: {elapsed:.2f}s")

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

        return JSONResponse({
            "audio": audio_base64,
            "sample_rate": sr,
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voices/design-save")
async def design_voice_save(
    name: str = Form(..., description="声音名称"),
    language: str = Form("Chinese", description="语言"),
    instruct: str = Form(..., description="声音描述"),
    text: str = Form(..., description="试听文本"),
    audio_base64: str = Form(..., description="预览生成的音频 base64"),
):
    """
    保存设计声音到声音库

    用预览生成的音频作为参考音频，提取 clone prompt 缓存
    """
    import base64
    import uuid

    if not name.strip():
        raise HTTPException(status_code=400, detail="声音名称不能为空")

    # 解码 base64 音频
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="音频数据无效")

    # 生成唯一 ID
    voice_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)

    try:
        # 保存音频文件
        audio_path = voice_dir / "reference.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 保存元数据（含声音描述）
        meta = {
            "name": name.strip(),
            "language": language,
            "ref_text": text,
            "audio_file": "reference.wav",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "design",
            "instruct": instruct,
        }
        with open(voice_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 如果克隆模型已加载，预生成 voice prompt
        prompt_cached = False
        if model_status["clone"] == "loaded":
            try:
                get_or_create_voice_prompt(voice_id, str(audio_path), text)
                prompt_cached = True
            except Exception as e:
                print(f"[警告] 预生成 prompt 失败: {e}")

        return {
            "success": True,
            "voice_id": voice_id,
            "name": name.strip(),
            "prompt_cached": prompt_cached,
            "message": "声音保存成功" + ("（已缓存）" if prompt_cached else "")
        }

    except Exception as e:
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voices/save")
async def save_voice(
    name: str = Form(..., description="声音名称"),
    language: str = Form("Chinese", description="语言"),
    ref_text: str = Form("", description="参考音频的文字内容"),
    audio: UploadFile = File(..., description="参考音频文件"),
):
    """
    保存克隆声音

    将参考音频和相关信息保存，以便后续直接使用
    """
    import time
    import uuid

    if not name.strip():
        raise HTTPException(status_code=400, detail="声音名称不能为空")

    # 生成唯一ID
    voice_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    voice_dir = VOICES_DIR / voice_id
    voice_dir.mkdir(exist_ok=True)

    try:
        # 保存音频文件
        audio_content = await audio.read()
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"
        audio_path = voice_dir / f"reference{suffix}"
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # 保存元数据
        meta = {
            "name": name.strip(),
            "language": language,
            "ref_text": ref_text,
            "audio_file": f"reference{suffix}",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(voice_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 如果克隆模型已加载，预生成 voice prompt
        prompt_cached = False
        if model_status["clone"] == "loaded":
            try:
                get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)
                prompt_cached = True
            except Exception as e:
                print(f"[警告] 预生成 prompt 失败: {e}")

        return {
            "success": True,
            "voice_id": voice_id,
            "name": name.strip(),
            "prompt_cached": prompt_cached,
            "message": "声音保存成功" + ("（已缓存）" if prompt_cached else "")
        }

    except Exception as e:
        # 清理失败的目录
        if voice_dir.exists():
            shutil.rmtree(voice_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """删除已保存的克隆声音"""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="声音不存在")

    try:
        shutil.rmtree(voice_dir)
        # 清除缓存
        if voice_id in saved_voice_prompts:
            del saved_voice_prompts[voice_id]
        return {"success": True, "message": "声音已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voices/{voice_id}/tts")
async def tts_with_saved_voice(
    voice_id: str,
    text: str = Form(..., description="要合成的文本"),
    language: str = Form(None, description="语言（可选，默认使用保存时的语言）"),
):
    """
    使用已保存的克隆声音生成语音
    """
    # 检查模型是否加载
    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "clone", "status": model_status["clone"]})

    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="声音不存在")

    meta_file = voice_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="声音元数据不存在")

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        audio_path = voice_dir / meta["audio_file"]
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="参考音频文件不存在")

        # 使用保存时的语言，除非指定了新语言
        use_language = language if language else meta.get("language", "Chinese")
        ref_text = meta.get("ref_text", "")

        print(f"[声音库] 开始生成: {len(text)} 字 | 声音: {meta.get('name', voice_id)} | 语言: {use_language}")
        start_time = time.time()

        # 获取或创建缓存的 voice prompt
        voice_prompt = get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)

        wavs, sr = model_clone.generate_voice_clone(
            text=text,
            language=use_language,
            voice_clone_prompt=[voice_prompt],
        )

        stats = get_generation_stats(text, start_time, "声音库")

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={voice_id}_output.wav",
                "X-Char-Count": str(stats["char_count"]),
                "X-Elapsed": str(stats["elapsed"]),
                "X-Avg-Per-Char": str(stats["avg_per_char"]),
                "Access-Control-Expose-Headers": "X-Char-Count, X-Elapsed, X-Avg-Per-Char",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices/{voice_id}/tts/progress")
async def tts_with_saved_voice_progress(
    request: Request,
    voice_id: str,
    text: str = Query(..., description="要合成的文本"),
    language: str = Query(None, description="语言（可选，默认使用保存时的语言）"),
):
    """
    使用已保存的克隆声音生成语音（带进度）

    - SSE 推送进度: {"progress": {"current": 1, "total": 10, "percent": 10}}
    - 最后推送完整音频: {"done": true, "audio": "base64..."}
    - 支持客户端断开时停止生成
    """
    import base64
    import asyncio

    # 检查模型是否加载
    if model_status["clone"] != "loaded":
        raise HTTPException(status_code=503, detail={"error": "model_not_loaded", "model": "clone", "status": model_status["clone"]})

    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="声音不存在")

    meta_file = voice_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="声音元数据不存在")

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        audio_path = voice_dir / meta["audio_file"]
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="参考音频文件不存在")

        # 使用保存时的语言，除非指定了新语言
        use_language = language if language else meta.get("language", "Chinese")
        ref_text = meta.get("ref_text", "")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def generate_with_progress():
        completed = 0
        total = 0
        try:
            # 分段落，再分句
            paragraphs = split_text_to_paragraphs(text, max_chars=300)
            all_sentences = []
            para_sent_map = []
            for para in paragraphs:
                start = len(all_sentences)
                sents = split_text_to_sentences(para, min_length=10)
                all_sentences.extend(sents)
                para_sent_map.append((start, len(all_sentences)))

            total = len(all_sentences)
            total_chars = len(text)
            num_paragraphs = len(paragraphs)

            print(f"[声音库进度] 开始生成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 声音: {meta.get('name', voice_id)}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars, 'paragraphs': num_paragraphs})}\n\n"

            # 获取或创建缓存的 voice prompt
            voice_prompt = get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)

            # 收集所有音频和字幕时间
            all_audio = []
            sentence_audios_b64 = []
            subtitles = []
            current_time = 0.0
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for p_idx, paragraph in enumerate(paragraphs):
                sent_start, sent_end = para_sent_map[p_idx]
                para_sentences = all_sentences[sent_start:sent_end]

                # 逐句生成
                for c_idx, sentence in enumerate(para_sentences):
                    if await request.is_disconnected():
                        print(f"[声音库进度] 客户端已断开，停止生成 (句 {sent_start+c_idx}/{total})")
                        return

                    wavs, sr = await loop.run_in_executor(
                        None,
                        lambda s=sentence, vp=voice_prompt: model_clone.generate_voice_clone(
                            text=s,
                            language=use_language,
                            voice_clone_prompt=[vp],
                        )
                    )

                    sample_rate = sr
                    chunk = wavs[0]
                    all_audio.append(chunk)

                    sent_buf = io.BytesIO()
                    sf.write(sent_buf, chunk, sr, format='WAV')
                    sent_buf.seek(0)
                    sentence_audios_b64.append(base64.b64encode(sent_buf.read()).decode('utf-8'))

                    duration = len(chunk) / sr
                    subtitles.append({"text": sentence, "start": round(current_time, 3), "end": round(current_time + duration, 3)})
                    current_time += duration

                    completed = sent_start + c_idx + 1
                    progress_data = {
                        "progress": {
                            "current": completed,
                            "total": total,
                            "percent": round(completed / total * 100),
                            "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                            "paragraph": p_idx + 1,
                            "total_paragraphs": num_paragraphs,
                        }
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"

                print(f"[声音库进度] 段落 {p_idx+1}/{num_paragraphs} 完成 (句 {sent_start+1}-{sent_end}/{total}): {paragraph[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[声音库进度] 全部完成: {total_chars} 字 | {total} 句 | {num_paragraphs} 段 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

            # 编码为 base64
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, merged_audio, sample_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            # 发送完成信号和音频
            done_data = {
                "done": True,
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "subtitles": subtitles,
                "sentence_audios": sentence_audios_b64,
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            print(f"[声音库进度] 生成被取消 ({completed}/{total})")
        except GeneratorExit:
            print(f"[声音库进度] 客户端断开连接 ({completed}/{total})")
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        generate_with_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/voices/{voice_id}/preview")
async def preview_saved_voice(voice_id: str):
    """获取已保存声音的参考音频（用于预览）"""
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail="声音不存在")

    meta_file = voice_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="声音元数据不存在")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    audio_path = voice_dir / meta["audio_file"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="参考音频文件不存在")

    return FileResponse(audio_path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
