import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

# 使用 CustomVoice 模型
model_path = "./models/Qwen3-TTS-0.6B-CustomVoice"

print("正在加载模型...")
model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 测试中文 (vivian 是中文说话人)
print("生成中文语音...")
wavs, sr = model.generate_custom_voice(
    text="你好，我是Qwen语音合成模型，很高兴认识你！",
    language="Chinese",
    speaker="vivian",
)
sf.write("output_zh.wav", wavs[0], sr)
print("中文语音已保存到 output_zh.wav")

# 测试英文 (aiden 是英文说话人)
print("生成英文语音...")
wavs, sr = model.generate_custom_voice(
    text="Hello, I am Qwen text to speech model. Nice to meet you!",
    language="English",
    speaker="aiden",
)
sf.write("output_en.wav", wavs[0], sr)
print("英文语音已保存到 output_en.wav")

print("完成！音频文件已生成。")
