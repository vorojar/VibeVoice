import io
import os
import json
import torch
import tempfile
import time
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil


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

app = FastAPI(title="Qwen3-TTS API", version="1.0.0")

# 添加 CORS 支持，允许 Tauri 应用访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境可以限制）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
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


def save_voice_prompt(voice_id: str, prompt_item):
    """将 VoiceClonePromptItem 保存到磁盘"""
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
    device_map, dtype = get_device_config()
    print(f"正在加载 {config['name']} 模型... (设备: {device_map}, 精度: {dtype})")

    try:
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
        "service": "Qwen3-TTS API",
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
    # 匹配一个或多个连续的中英文标点
    pattern = r'([。！？；.!?;]+|\n)'
    parts = re.split(pattern, text)

    # 先按标点分成基本句子
    raw_sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            if current.strip():
                raw_sentences.append(current.strip())
            current = ""
    if current.strip():
        raw_sentences.append(current.strip())

    # 如果没有分割出句子，返回整个文本
    if not raw_sentences:
        return [text] if text.strip() else []

    # 合并过短的句子
    merged = []
    buffer = ""
    for sentence in raw_sentences:
        buffer += sentence
        if len(buffer) >= min_length:
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
            # 分句
            sentences = split_text_to_sentences(text, min_length=10)
            total = len(sentences)
            total_chars = len(text)

            print(f"[TTS进度] 开始生成: {total_chars} 字 | {total} 句 | 说话人: {speaker}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars})}\n\n"

            # 收集所有音频
            all_audio = []
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for i, sentence in enumerate(sentences):
                # 检查客户端是否断开
                if await request.is_disconnected():
                    print(f"[TTS进度] 客户端已断开，停止生成 ({i}/{total})")
                    return

                # 在线程池中运行阻塞的模型推理
                wavs, sr = await loop.run_in_executor(
                    None,
                    lambda s=sentence: model_custom.generate_custom_voice(
                        text=s,
                        language=language,
                        speaker=speaker,
                        instruct=instruct,
                    )
                )

                # 生成后再次检查断开
                if await request.is_disconnected():
                    print(f"[TTS进度] 客户端已断开，停止生成 ({i+1}/{total})")
                    return

                all_audio.append(wavs[0])
                sample_rate = sr
                completed = i + 1

                # 推送进度
                progress_data = {
                    "progress": {
                        "current": i + 1,
                        "total": total,
                        "percent": round((i + 1) / total * 100),
                        "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                    }
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                print(f"[TTS进度] {i+1}/{total} 完成: {sentence[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[TTS进度] 全部完成: {total_chars} 字 | {total} 句 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

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
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data)}\n\n"

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
            # 分句
            sentences = split_text_to_sentences(text, min_length=10)
            total = len(sentences)
            total_chars = len(text)

            print(f"[设计进度] 开始生成: {total_chars} 字 | {total} 句 | 语言: {language}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars})}\n\n"

            # 收集所有音频
            all_audio = []
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for i, sentence in enumerate(sentences):
                # 检查客户端是否断开
                if await request.is_disconnected():
                    print(f"[设计进度] 客户端已断开，停止生成 ({i}/{total})")
                    return

                # 在线程池中运行阻塞的模型推理
                wavs, sr = await loop.run_in_executor(
                    None,
                    lambda s=sentence: model_design.generate_voice_design(
                        text=s,
                        language=language,
                        instruct=instruct,
                    )
                )

                # 生成后再次检查断开
                if await request.is_disconnected():
                    print(f"[设计进度] 客户端已断开，停止生成 ({i+1}/{total})")
                    return

                all_audio.append(wavs[0])
                sample_rate = sr
                completed = i + 1

                # 推送进度
                progress_data = {
                    "progress": {
                        "current": i + 1,
                        "total": total,
                        "percent": round((i + 1) / total * 100),
                        "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                    }
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                print(f"[设计进度] {i+1}/{total} 完成: {sentence[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[设计进度] 全部完成: {total_chars} 字 | {total} 句 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

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
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data)}\n\n"

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
            # 分句
            sentences = split_text_to_sentences(text, min_length=10)
            total = len(sentences)
            total_chars = len(text)

            print(f"[声音库进度] 开始生成: {total_chars} 字 | {total} 句 | 声音: {meta.get('name', voice_id)}")
            start_time = time.time()

            # 发送开始信号
            yield f"data: {json.dumps({'started': True, 'total': total, 'total_chars': total_chars})}\n\n"

            # 获取或创建缓存的 voice prompt
            voice_prompt = get_or_create_voice_prompt(voice_id, str(audio_path), ref_text)

            # 收集所有音频
            all_audio = []
            sample_rate = SAMPLE_RATE
            loop = asyncio.get_event_loop()

            for i, sentence in enumerate(sentences):
                # 检查客户端是否断开
                if await request.is_disconnected():
                    print(f"[声音库进度] 客户端已断开，停止生成 ({i}/{total})")
                    return

                # 在线程池中运行阻塞的模型推理
                wavs, sr = await loop.run_in_executor(
                    None,
                    lambda s=sentence, vp=voice_prompt: model_clone.generate_voice_clone(
                        text=s,
                        language=use_language,
                        voice_clone_prompt=[vp],
                    )
                )

                # 生成后再次检查断开
                if await request.is_disconnected():
                    print(f"[声音库进度] 客户端已断开，停止生成 ({i+1}/{total})")
                    return

                all_audio.append(wavs[0])
                sample_rate = sr
                completed = i + 1

                # 推送进度
                progress_data = {
                    "progress": {
                        "current": i + 1,
                        "total": total,
                        "percent": round((i + 1) / total * 100),
                        "sentence": sentence[:20] + "..." if len(sentence) > 20 else sentence,
                    }
                }
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                print(f"[声音库进度] {i+1}/{total} 完成: {sentence[:30]}...")

            # 合并音频
            merged_audio = np.concatenate(all_audio)

            # 计算统计
            elapsed = time.time() - start_time
            avg_per_char = elapsed / total_chars if total_chars > 0 else 0
            print(f"[声音库进度] 全部完成: {total_chars} 字 | {total} 句 | 用时 {elapsed:.2f}s | 平均 {avg_per_char:.3f}s/字")

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
                "stats": {
                    "char_count": total_chars,
                    "sentence_count": total,
                    "elapsed": round(elapsed, 2),
                    "avg_per_char": round(avg_per_char, 3),
                }
            }
            yield f"data: {json.dumps(done_data)}\n\n"

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
