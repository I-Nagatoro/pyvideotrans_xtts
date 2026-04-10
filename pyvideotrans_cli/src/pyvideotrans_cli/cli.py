#!/usr/bin/env python3
"""
pyvideotrans-cli - 精简版视频翻译 CLI 工具

完整 Pipeline:
1. 视频→音频提取 (ffmpeg)
2. 语音识别 (Whisper)
3. 文本翻译 (本地 LLM API)
4. TTS 配音 (Qwen TTS API/Local)
5. 音视频合并 (ffmpeg)
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="pyvideotrans-cli - 精简版视频翻译工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 核心参数
    parser.add_argument("-i", "--input", required=True, help="输入视频文件路径")
    parser.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    parser.add_argument("--mode", choices=["full", "transcribe", "translate", "tts", "merge"], 
                        default="full", help="运行模式")
    
    # 转录参数
    transcribe_group = parser.add_argument_group("Transcription Options")
    transcribe_group.add_argument("--whisper-model", default="tiny", 
                                  choices=["tiny", "base", "small", "medium", "large-v3"],
                                  help="Whisper 模型大小")
    transcribe_group.add_argument("--source-lang", default="auto", help="源语言代码")
    transcribe_group.add_argument("--cuda", action="store_true", help="使用 CUDA 加速")
    
    # 翻译参数
    translate_group = parser.add_argument_group("Translation Options")
    translate_group.add_argument("--target-lang", required="--mode" in sys.argv and "translate" in sys.argv or "--mode" in sys.argv and "full" in sys.argv,
                                 help="目标语言代码 (如：en, zh, ja)")
    translate_group.add_argument("--transformers", action="store_true", 
                                 help="Использовать локальную модель Transformers (NLLB) вместо LLM API")
    translate_group.add_argument("--llm-api", default=None, 
                                 help="LLM API 地址 (如：https://api.openai.com/v1, https://api.deepseek.com/v1)")
    translate_group.add_argument("--llm-key", required=not ("--transformers" in sys.argv), help="LLM API Key (не требуется при использовании --transformers)")
    translate_group.add_argument("--llm-model", default="gpt-3.5-turbo", help="LLM 模型名称")
    translate_group.add_argument("--llm-provider", choices=["openai", "deepseek", "qwen", "custom"], default="openai",
                                 help="LLM 提供商类型")
    
    # TTS 参数
    tts_group = parser.add_argument_group("TTS Options")
    tts_group.add_argument("--qwen-api-key", help="Qwen TTS API Key (必选用于 API 模式)")
    tts_group.add_argument("--qwen-model", default="qwen3-tts-flash", help="Qwen TTS 模型")
    tts_group.add_argument("--qwen-voice", default="Cherry", help="Qwen TTS 音色")
    tts_group.add_argument("--qwen-local", action="store_true", help="使用本地 Qwen TTS (需额外实现)")
    tts_group.add_argument("--qwen-local-model", default="1.7B", help="本地 Qwen TTS 模型大小")
    
    # 合并参数
    merge_group = parser.add_argument_group("Merge Options")
    merge_group.add_argument("--keep-original-audio", action="store_true", 
                             help="保留原始音频 (混合新旧音频)")
    merge_group.add_argument("--add-subtitles", action="store_true", help="添加字幕到视频")
    merge_group.add_argument("--hardsub", action="store_true", help="硬字幕 (烧录), 否则软字幕")
    
    args = parser.parse_args()
    
    # Преобразуем output_dir в Path для удобной работы с путями
    args.output_dir = Path(args.output_dir)
    
    #验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == "transcribe":
            run_transcribe(args)
        elif args.mode == "translate":
            run_translate(args)
        elif args.mode == "tts":
            run_tts(args)
        elif args.mode == "merge":
            run_merge(args)
        elif args.mode == "full":
            run_full_pipeline(args)
        
        logger.info("✅ Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


def run_transcribe(args):
    """仅执行语音转录"""
    from video import VideoProcessor
    from transcribe import Transcriber
    
    # 1. 提取音频
    processor = VideoProcessor(str(args.input), str(args.output_dir))
    audio_file = processor.extract_audio()
    
    # 2. 语音识别
    transcriber = Transcriber(
        audio_file=audio_file,
        model_name=args.whisper_model,
        language=args.source_lang,
        is_cuda=args.cuda
    )
    
    subtitles = transcriber.transcribe()
    srt_path = args.output_dir / "source.srt"
    srt_file = transcriber.save_to_srt(subtitles, str(srt_path))
    
    logger.info(f"Transcription saved to: {srt_file}")
    return srt_file


def run_translate(args):
    """仅执行字幕翻译"""
    from translate import Translator
    
    # 读取源字幕
    source_srt_path = args.output_dir / "source.srt"
    source_srt = str(source_srt_path)
    if not source_srt_path.exists():
        # 尝试从视频中提取并转录
        logger.info("Source subtitles not found, running transcription first...")
        from video import VideoProcessor
        from transcribe import Transcriber
        
        processor = VideoProcessor(str(args.input), str(args.output_dir))
        audio_file = processor.extract_audio()
        transcriber = Transcriber(audio_file=audio_file, model_name=args.whisper_model)
        subtitles = transcriber.transcribe()
        source_srt = transcriber.save_to_srt(subtitles, str(source_srt_path))
    
    # 解析 SRT
    subtitles = parse_srt(source_srt)
    
    # Перевод через Transformers или LLM API
    if args.transformers:
        translator = Translator(
            subtitles=subtitles,
            target_language=args.target_lang,
            use_transformers=True
        )
    else:
        # 设置 API URL 基于提供商
        api_url = args.llm_api
        if not api_url:
            if args.llm_provider == "openai":
                api_url = "https://api.openai.com/v1"
            elif args.llm_provider == "deepseek":
                api_url = "https://api.deepseek.com/v1"
            elif args.llm_provider == "qwen":
                api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                raise ValueError("--llm-api is required for custom provider")
        
        translator = Translator(
            subtitles=subtitles,
            target_language=args.target_lang,
            api_url=api_url,
            api_key=args.llm_key,
            model_name=args.llm_model
        )
    
    translated = translator.translate()
    output_srt_path = args.output_dir / f"{args.target_lang}.srt"
    output_srt = translator.save_to_srt(translated, str(output_srt_path))
    
    logger.info(f"Translation saved to: {output_srt}")
    return output_srt


def run_tts(args):
    """仅执行 TTS"""
    from tts import QwenTTS, QwenTTSLocal
    
    # 读取目标语言字幕
    target_srt_path = args.output_dir / f"{args.target_lang}.srt"
    target_srt = str(target_srt_path)
    if not target_srt_path.exists():
        raise FileNotFoundError(f"Target subtitles not found: {target_srt_path}")
    
    subtitles = parse_srt(target_srt)
    
    if args.qwen_local:
        tts = QwenTTSLocal(
            subtitles=subtitles,
            target_language=args.target_lang,
            model_name=args.qwen_local_model,
            output_dir=str(args.output_dir / "tts"),
            device="cuda" if args.cuda else "cpu",
            torch_dtype="float16" if args.cuda else "float32"
        )
    else:
        if not args.qwen_api_key:
            raise ValueError("--qwen-api-key is required for Qwen TTS API mode")
        
        tts = QwenTTS(
            subtitles=subtitles,
            target_language=args.target_lang,
            api_key=args.qwen_api_key,
            model=args.qwen_model,
            voice=args.qwen_voice,
            output_dir=str(args.output_dir / "tts")
        )
    
    result = tts.synthesize()
    
    # 合并音频
    merged_audio_path = args.output_dir / "dubbed.wav"
    merged_audio = tts.merge_audio(str(merged_audio_path))
    logger.info(f"Merged audio saved to: {merged_audio}")
    
    return merged_audio


def run_merge(args):
    """仅执行音视频合并"""
    from video import VideoProcessor
    
    processor = VideoProcessor(str(args.input), str(args.output_dir))
    
    dubbed_audio_path = args.output_dir / "dubbed.wav"
    dubbed_audio = str(dubbed_audio_path)
    if not dubbed_audio_path.exists():
        raise FileNotFoundError(f"Dubbed audio not found: {dubbed_audio_path}")
    
    output_video = processor.merge_audio_video(
        dubbed_audio,
        original_audio=args.keep_original_audio
    )
    
    if args.add_subtitles:
        subtitle_file_path = args.output_dir / f"{args.target_lang}.srt"
        if subtitle_file_path.exists():
            output_video = processor.add_subtitles(
                str(subtitle_file_path),
                hardsub=args.hardsub
            )
    
    logger.info(f"Final video saved to: {output_video}")
    return output_video


def run_full_pipeline(args):
    """执行完整 pipeline"""
    logger.info("🚀 Starting full pipeline...")
    
    # Step 1: 转录
    logger.info("\n📝 Step 1/4: Transcribing...")
    run_transcribe(args)
    
    # Step 2: 翻译
    logger.info("\n🌐 Step 2/4: Translating...")
    run_translate(args)
    
    # Step 3: TTS
    logger.info("\n🔊 Step 3/4: Generating TTS...")
    run_tts(args)
    
    # Step 4: 合并
    logger.info("\n🎬 Step 4/4: Merging...")
    run_merge(args)
    
    logger.info("\n✅ Full pipeline completed!")


def parse_srt(srt_file: str):
    """解析 SRT 文件为字典列表"""
    subtitles = []
    
    with open(srt_file, encoding='utf-8') as f:
        content = f.read()
    
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # 时间行
        time_line = lines[1]
        start_str, end_str = time_line.split(' --> ')
        
        def parse_time(t: str) -> float:
            parts = t.replace(',', '.').split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        
        text = '\n'.join(lines[2:])
        
        subtitles.append({
            "start_time": parse_time(start_str),
            "end_time": parse_time(end_str),
            "text": text
        })
    
    return subtitles


if __name__ == "__main__":
    main()
