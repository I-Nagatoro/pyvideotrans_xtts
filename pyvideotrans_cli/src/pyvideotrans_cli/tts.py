"""
TTS 配音模块 - 使用 Qwen TTS (API / Local)
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QwenTTS:
    """Qwen TTS 配音器 (API 版本)"""
    subtitles: List[Dict]
    target_language: str = "en"
    api_key: Optional[str] = None
    model: str = "qwen3-tts-flash"
    voice: str = "Cherry"
    output_dir: str = "./output"
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("Qwen TTS API key is required")
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def synthesize(self) -> List[Dict]:
        """
        执行 TTS，返回带音频文件路径的字幕列表
        Returns:
            List[Dict]: [{"start_time": 0.0, "end_time": 1.5, "text": "...", "filename": "path/to/audio.wav"}]
        """
        try:
            import dashscope
            import requests
            
            logger.info(f"Using Qwen TTS model: {self.model}")
            
            for i, sub in enumerate(self.subtitles):
                if not sub.get("text", "").strip():
                    continue
                
                output_file = self.output_path / f"audio_{i:04d}.wav"
                
                # 调用 Qwen TTS API
                response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                    model=self.model,
                    api_key=self.api_key,
                    text=sub["text"],
                    voice=self.voice
                )
                
                if response is None:
                    raise RuntimeError("API call returned None response")
                
                if not hasattr(response, 'output') or response.output is None:
                    raise RuntimeError(f"TTS API error: {response.message if hasattr(response, 'message') else str(response)}")
                
                # 下载音频文件
                audio_url = response.output.audio.get("url")
                if not audio_url:
                    raise RuntimeError("No audio URL in response")
                
                res = requests.get(audio_url)
                res.raise_for_status()
                
                # 保存为 WAV
                with open(output_file, 'wb') as f:
                    f.write(res.content)
                
                sub["filename"] = str(output_file)
                logger.info(f"Generated audio for segment {i+1}/{len(self.subtitles)}")
            
            return self.subtitles
            
        except ImportError as e:
            logger.error(f"dashscope not installed: {e}")
            raise RuntimeError("Please install dashscope: pip install dashscope")
    
    def merge_audio(self, output_path: str) -> str:
        """合并所有音频片段为一个文件"""
        from pydub import AudioSegment
        
        merged = AudioSegment.empty()
        
        for sub in self.subtitles:
            if "filename" in sub and Path(sub["filename"]).exists():
                audio = AudioSegment.from_wav(sub["filename"])
                # 根据时间戳添加静音间隔
                silence_duration = int(sub["start_time"] * 1000) - len(merged)
                if silence_duration > 0:
                    merged += AudioSegment.silent(duration=silence_duration)
                merged += audio
        
        output_file = Path(output_path)
        merged.export(str(output_file), format="wav")
        logger.info(f"Merged audio saved to: {output_file}")
        return str(output_file)


@dataclass
class QwenTTSLocal:
    """Qwen TTS Local 版本 - 使用 transformers 进行本地推理"""
    subtitles: List[Dict]
    target_language: str = "en"
    model_name: str = "Qwen/Qwen3-TTS-0.7B"
    output_dir: str = "./output"
    device: str = "cuda"  # "cuda" или "cpu"
    torch_dtype: str = "float16"  # "float16" или "float32"
    
    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.speaker_embedder = None
        
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        if self.model is not None:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLTTS
            import torch
            from accelerate import load_checkpoint_and_dispatch
            
            logger.info(f"Загрузка локальной модели Qwen TTS: {self.model_name}")
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Определение типа данных
            if self.torch_dtype == "float16" and self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # Загрузка модели
            self.model = AutoModelForCausalLTTS.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            self.model.eval()
            logger.info("Модель Qwen TTS успешно загружена")
            
        except ImportError as e:
            logger.error(f"Необходимые библиотеки не установлены: {e}")
            raise RuntimeError("Установите зависимости: pip install transformers accelerate torch torchaudio")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def _synthesize_segment(self, text: str, output_file: Path) -> str:
        """Синтез одного сегмента текста в аудио"""
        import torch
        import soundfile as sf
        
        if self.model is None:
            self._load_model()
        
        # Подготовка входных данных для модели
        # Формат промпта зависит от конкретной версии Qwen TTS
        prompt = f"<|text|>{text}<|audio|>"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Генерация аудио
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Извлечение аудио из выходов модели
        # Конкретный способ зависит от архитектуры модели
        if hasattr(outputs, 'audio'):
            audio_data = outputs.audio.cpu().numpy()
        else:
            # Если модель возвращает logits, нужно применить декодер
            # Это упрощенная версия - реальная реализация зависит от модели
            logger.warning("Используется упрощенная экстракция аудио")
            audio_data = outputs.cpu().numpy().flatten()
        
        # Нормализация аудио
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Нормализация к диапазону [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Сохранение аудиофайла
        sf.write(str(output_file), audio_data, samplerate=24000)
        
        return str(output_file)
    
    def synthesize(self) -> List[Dict]:
        """
        Использование локальной модели Qwen TTS для синтеза
        Returns:
            List[Dict]: [{\"start_time\": 0.0, \"end_time\": 1.5, \"text\": \"...\", \"filename\": \"path/to/audio.wav\"}]
        """
        import numpy as np
        
        logger.info("Запуск локального Qwen TTS синтеза")
        logger.warning("Для работы требуется модель Qwen3-TTS от HuggingFace")
        
        for i, sub in enumerate(self.subtitles):
            if not sub.get("text", "").strip():
                continue
            
            output_file = self.output_path / f"audio_{i:04d}.wav"
            
            try:
                # Синтез аудио для текущего сегмента
                audio_path = self._synthesize_segment(sub["text"], output_file)
                sub["filename"] = audio_path
                logger.info(f"Сгенерировано аудио для сегмента {i+1}/{len(self.subtitles)}: {audio_path}")
                
            except Exception as e:
                logger.error(f"Ошибка синтеза для сегмента {i}: {e}")
                raise
        
        return self.subtitles
    
    def merge_audio(self, output_path: str) -> str:
        """Объединение всех аудиофрагментов в один файл"""
        from pydub import AudioSegment
        
        merged = AudioSegment.empty()
        
        for sub in self.subtitles:
            if "filename" in sub and Path(sub["filename"]).exists():
                audio = AudioSegment.from_wav(sub["filename"])
                # Добавление тишины согласно таймингам
                silence_duration = int(sub["start_time"] * 1000) - len(merged)
                if silence_duration > 0:
                    merged += AudioSegment.silent(duration=silence_duration)
                merged += audio
        
        output_file = Path(output_path)
        merged.export(str(output_file), format="wav")
        logger.info(f"Объединенное аудио сохранено: {output_file}")
        return str(output_file)
