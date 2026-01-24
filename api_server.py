import io
import os
import json
import torch
import tempfile
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil

app = FastAPI(title="Qwen3-TTS API", version="1.0.0")

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

# 已保存的克隆声音缓存
saved_voice_prompts = {}

# 支持的说话人
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

# 支持的语言
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


def load_model_sync(model_type: str):
    """同步加载指定模型"""
    global model_custom, model_clone, model_design, model_status

    if model_status[model_type] != "unloaded":
        return

    model_status[model_type] = "loading"
    config = MODEL_CONFIGS[model_type]
    print(f"正在加载 {config['name']} 模型...")

    try:
        model = Qwen3TTSModel.from_pretrained(
            config["path"],
            device_map="cuda:0",
            dtype=torch.bfloat16,
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
        wavs, sr = model_custom.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
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

def split_text_to_sentences(text: str) -> list:
    """将文本按标点符号分割成句子"""
    # 中英文标点
    pattern = r'([。！？；\n.!?;])'
    parts = re.split(pattern, text)

    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    # 如果没有分割出句子，返回整个文本
    if not sentences:
        sentences = [text]

    return sentences


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

    async def generate_audio_stream():
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
        use_x_vector_only = not ref_text
        wavs, sr = model_clone.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=temp_file.name,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=use_x_vector_only,
        )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=clone_output.wav"}
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
        wavs, sr = model_design.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=design_output.wav"}
        )

    except Exception as e:
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
                    voices.append({
                        "id": voice_dir.name,
                        "name": meta.get("name", voice_dir.name),
                        "language": meta.get("language", "Chinese"),
                        "created_at": meta.get("created_at", ""),
                    })
    return {"voices": voices}


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

        return {
            "success": True,
            "voice_id": voice_id,
            "name": name.strip(),
            "message": "声音保存成功"
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
        use_x_vector_only = not ref_text

        wavs, sr = model_clone.generate_voice_clone(
            text=text,
            language=use_language,
            ref_audio=str(audio_path),
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=use_x_vector_only,
        )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format='WAV')
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={voice_id}_output.wav"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
