"""
视频处理模块 - 使用 ffmpeg 进行音视频处理
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessor:
    """视频处理器"""
    video_file: str
    output_dir: str = "./output"
    
    def __post_init__(self):
        self.video_path = Path(self.video_file)
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
    
    def extract_audio(self, output_path: Optional[str] = None) -> str:
        """从视频中提取音频"""
        if output_path is None:
            output_path = self.output_path / f"{self.video_path.stem}.wav"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Extracting audio from {self.video_path}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        logger.info(f"Audio extracted to: {output_path}")
        return str(output_path)
    
    def get_video_info(self) -> Dict:
        """获取视频信息"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(self.video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe error: {result.stderr}")
        
        import json
        info = json.loads(result.stdout)
        
        video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
        
        return {
            "duration": float(info.get("format", {}).get("duration", 0)),
            "width": video_stream.get("width", 0) if video_stream else 0,
            "height": video_stream.get("height", 0) if video_stream else 0,
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
            "video_codec": video_stream.get("codec_name", "") if video_stream else "",
            "audio_codec": audio_stream.get("codec_name", "") if audio_stream else ""
        }
    
    def merge_audio_video(self, audio_file: str, output_path: Optional[str] = None, 
                          original_audio: bool = False, volume: float = 1.0) -> str:
        """
        将音频与视频合并
        
        Args:
            audio_file: 新的音频文件路径
            output_path: 输出文件路径
            original_audio: 是否保留原始音频 (混合)
            volume: 新音频的音量 (1.0=正常)
        """
        if output_path is None:
            output_path = self.output_path / f"{self.video_path.stem}_dubbed.mp4"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Merging audio {audio_file} with video {self.video_path}")
        
        if original_audio:
            # 混合新旧音频
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", audio_file,
                "-filter_complex", f"[1:a]volume={volume}[new];[0:a][new]amix=inputs=2:duration=first[aout]",
                "-c:v", "copy",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:a", "aac",
                str(output_path)
            ]
        else:
            # 替换音频
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", audio_file,
                "-c:v", "copy",
                "-map", "0:v",
                "-map", "1:a",
                "-c:a", "aac",
                "-shortest",
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        logger.info(f"Video with audio saved to: {output_path}")
        return str(output_path)
    
    def add_subtitles(self, subtitle_file: str, output_path: Optional[str] = None, 
                      hardsub: bool = True, language: str = "chi") -> str:
        """
        添加字幕到视频
        
        Args:
            subtitle_file: SRT 字幕文件路径
            output_path: 输出文件路径
            hardsub: True=硬字幕 (烧录), False=软字幕 (可选)
            language: 字幕语言代码 (软字幕时使用)
        """
        if output_path is None:
            suffix = "hardsub" if hardsub else "softsub"
            output_path = self.output_path / f"{self.video_path.stem}_{suffix}.mp4"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Adding subtitles from {subtitle_file}")
        
        if hardsub:
            # 硬字幕 - 烧录到视频
            # 转义路径中的特殊字符
            subtitle_path_escaped = str(Path(subtitle_file)).replace("'", "'\\''").replace(":", r"\:")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-vf", f"subtitles='{subtitle_path_escaped}'",
                "-c:a", "copy",
                str(output_path)
            ]
        else:
            # 软字幕 - 作为独立流
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", subtitle_file,
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language={language}",
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        logger.info(f"Video with subtitles saved to: {output_path}")
        return str(output_path)
