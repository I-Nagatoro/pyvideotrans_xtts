"""
精简版视频翻译 CLI 工具

支持的功能:
1. 视频→音频提取 (ffmpeg)
2. 语音识别 (Whisper - faster-whisper/openai-whisper)
3. 文本翻译 (本地 LLM API)
4. TTS 配音 (Qwen TTS API / Local)
5. 音视频合并

使用方法:
    # 完整流程：视频翻译
    pyvideotrans-cli -i video.mp4 --target-lang en --qwen-api-key YOUR_KEY
    
    # 仅转录
    pyvideotrans-cli -i video.mp4 --mode transcribe --output-dir ./output
    
    # 仅翻译字幕
    pyvideotrans-cli -i video.mp4 --mode translate --target-lang en --llm-api http://localhost:1234/v1
    
    # 仅 TTS
    pyvideotrans-cli -i video.mp4 --mode tts --target-lang en --qwen-api-key YOUR_KEY
"""

__version__ = "0.1.0"
__author__ = "Customized from pyvideotrans"
