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
SAMPLE_RATE = 24000

# 已保存的克隆声音缓存
saved_voice_prompts = {}

# 支持的说话人
SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]

# 支持的语言
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]


@app.on_event("startup")
async def load_model():
    global model_custom, model_clone

    print("正在加载 CustomVoice 模型...")
    model_custom = Qwen3TTSModel.from_pretrained(
        "./models/Qwen3-TTS-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("CustomVoice 模型加载完成！")

    print("正在加载 Base 模型（语音克隆）...")
    model_clone = Qwen3TTSModel.from_pretrained(
        "./models/Qwen3-TTS-0.6B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Base 模型加载完成！")


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
