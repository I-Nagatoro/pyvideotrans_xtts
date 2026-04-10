"""
语音转录模块 - 使用 Whisper 进行语音识别
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Transcriber:
    """Whisper 语音转录器"""
    audio_file: str
    model_name: str = "tiny"
    language: str = "auto"
    is_cuda: bool = False
    output_format: str = "srt"
    
    def __post_init__(self):
        self.audio_path = Path(self.audio_file)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")
    
    def transcribe(self) -> List[Dict]:
        """
        执行语音转录，返回字幕列表
        Returns:
            List[Dict]: [{"start_time": 0.0, "end_time": 1.5, "text": "Hello"}]
        """
        try:
            # 使用 faster-whisper (推荐)
            from faster_whisper import WhisperModel
            
            device = "cuda" if self.is_cuda else "cpu"
            compute_type = "float16" if self.is_cuda else "int8"
            
            logger.info(f"Loading Whisper model: {self.model_name} on {device}")
            model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
            
            lang_code = None if self.language == "auto" else self.language
            
            segments, info = model.transcribe(
                str(self.audio_path),
                language=lang_code,
                word_timestamps=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            logger.info(f"Detected language: {info.language}")
            
            result = []
            for segment in segments:
                result.append({
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "text": segment.text.strip()
                })
            
            return result
            
        except ImportError:
            # 降级到 openai-whisper
            logger.warning("faster-whisper not available, falling back to openai-whisper")
            return self._transcribe_openai()
    
    def _transcribe_openai(self) -> List[Dict]:
        """使用 openai-whisper 进行转录"""
        import whisper
        
        model = whisper.load_model(self.model_name)
        result = model.transcribe(
            str(self.audio_path),
            language=None if self.language == "auto" else self.language
        )
        
        return [
            {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result.get("segments", [])
        ]
    
    def save_to_srt(self, subtitles: List[Dict], output_path: str) -> str:
        """保存字幕为 SRT 格式"""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        srt_content = ""
        for i, sub in enumerate(subtitles, 1):
            srt_content += f"{i}\n"
            srt_content += f"{format_time(sub['start_time'])} --> {format_time(sub['end_time'])}\n"
            srt_content += f"{sub['text']}\n\n"
        
        output_file = Path(output_path)
        output_file.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT saved to: {output_file}")
        return str(output_file)
    
    def save_to_json(self, subtitles: List[Dict], output_path: str) -> str:
        """保存字幕为 JSON 格式"""
        output_file = Path(output_path)
        output_file.write_text(json.dumps(subtitles, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"JSON saved to: {output_file}")
        return str(output_file)
