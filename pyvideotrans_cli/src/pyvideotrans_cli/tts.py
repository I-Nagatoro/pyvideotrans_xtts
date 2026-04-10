"""
TTS 配音模块 - 使用 Qwen TTS (API / Local) и Coqui XTTS v2
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
        """Объединение всех аудиофрагментов в один файл с правильными таймингами"""
        from pydub import AudioSegment
        import numpy as np
        
        logger.info("Начало объединения аудиофрагментов...")
        
        # Сортируем субтитры по времени начала
        sorted_subs = sorted(self.subtitles, key=lambda x: x["start_time"])
        
        # Определяем общую длительность видео по последнему таймингу
        if sorted_subs:
            total_duration_ms = int(sorted_subs[-1]["end_time"] * 1000)
        else:
            total_duration_ms = 0
        
        # Создаём пустой аудиофайл нужной длительности
        merged = AudioSegment.silent(duration=total_duration_ms)
        
        for i, sub in enumerate(sorted_subs):
            if "filename" not in sub or not Path(sub["filename"]).exists():
                logger.warning(f"Аудиофайл для сегмента {i} не найден, пропускаем")
                continue
            
            try:
                audio = AudioSegment.from_wav(sub["filename"])
                
                # Обрезаем тишину в начале и конце аудио (очистка артефактов)
                audio = self._trim_silence(audio, silence_threshold=-50, min_silence_duration=100)
                
                # Рассчитываем позицию наложения
                start_position_ms = int(sub["start_time"] * 1000)
                
                # Обрезаем аудио если оно выходит за рамки end_time
                max_duration_ms = int(sub["end_time"] * 1000) - start_position_ms
                if len(audio) > max_duration_ms:
                    audio = audio[:max_duration_ms]
                    logger.warning(f"Сегмент {i} обрезан по таймингу")
                
                # Накладываем аудио на нужную позицию
                merged = merged.overlay(audio, position=start_position_ms)
                
                logger.debug(f"Сегмент {i}: начало={start_position_ms}мс, длительность={len(audio)}мс")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке сегмента {i}: {e}")
                continue
        
        # Экспортируем результат
        output_file = Path(output_path)
        merged.export(str(output_file), format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000"])
        logger.info(f"Объединенное аудио сохранено: {output_file}, общая длительность: {len(merged)}мс")
        return str(output_file)
    
    def _trim_silence(self, audio: "AudioSegment", silence_threshold: float = -50, min_silence_duration: int = 100) -> "AudioSegment":
        """
        Обрезает тишину в начале и конце аудио
        
        Args:
            audio: AudioSegment для обработки
            silence_threshold: Порог тишины в dBFS (по умолчанию -50)
            min_silence_duration: Минимальная длительность тишины в мс для обрезки
        
        Returns:
            AudioSegment с обрезанной тишиной
        """
        from pydub.silence import detect_leading_silence, detect_trailing_silence
        
        # Обрезаем тишину в начале
        leading_silence = detect_leading_silence(audio, silence_threshold=silence_threshold)
        if leading_silence > min_silence_duration:
            audio = audio[leading_silence:]
        
        # Обрезаем тишину в конце
        trailing_silence = detect_trailing_silence(audio, silence_threshold=silence_threshold)
        if trailing_silence > min_silence_duration:
            audio = audio[:-trailing_silence]
        
        return audio


@dataclass
class CoquiXTTS:
    """Coqui XTTS v2 - локальная модель TTS с поддержкой множественных языков"""
    subtitles: List[Dict]
    target_language: str = "en"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    output_dir: str = "./output"
    device: str = "cuda"  # "cuda" или "cpu"
    speaker_wav: Optional[str] = None  # Путь к примеру голоса для клонирования
    language: str = "en"  # Язык синтеза
    source_video: Optional[str] = None  # Исходное видео для извлечения голоса
    
    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        
        # Маппинг кодов языков для XTTS
        self.language_map = {
            "en": "en",
            "zh": "zh-cn",
            "fr": "fr",
            "de": "de",
            "es": "es",
            "it": "it",
            "pt": "pt",
            "pl": "pl",
            "tr": "tr",
            "ru": "ru",
            "nl": "nl",
            "cs": "cs",
            "ar": "ar",
            "hu": "hu",
            "ko": "ko",
            "ja": "ja",
            "hi": "hi",
        }
    
    def _extract_speaker_from_video(self, video_path: str) -> str:
        """Извлечение образца голоса из исходного видео (первые 10 секунд)"""
        import subprocess
        
        speaker_file = self.output_path / "speaker_sample.wav"
        
        logger.info(f"Извлечение образца голоса из видео: {video_path}")
        
        # Извлекаем первые 10 секунд аудио для использования в качестве образца голоса
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-t", "10",  # Первые 10 секунд
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",  # XTTS предпочитает 22050 Гц
            "-ac", "1",
            str(speaker_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error при извлечении голоса: {result.stderr}")
        
        logger.info(f"Образец голоса сохранён: {speaker_file}")
        return str(speaker_file)
    
    def _load_model(self):
        """Загрузка модели XTTS v2"""
        if self.model is not None:
            return
            
        try:
            from TTS.api import TTS
            import torch
            
            logger.info(f"Загрузка локальной модели Coqui XTTS v2: {self.model_name}")
            
            # Определение доступных устройств
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA не доступна, переключение на CPU")
                self.device = "cpu"
            
            # Загрузка модели
            self.model = TTS(model_name=self.model_name).to(self.device)
            
            logger.info("Модель Coqui XTTS v2 успешно загружена")
            
        except ImportError as e:
            logger.error(f"Необходимые библиотеки не установлены: {e}")
            raise RuntimeError("Установите зависимости: pip install TTS")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def _synthesize_segment(self, text: str, output_file: Path, speaker_wav: Optional[str] = None) -> str:
        """Синтез одного сегмента текста в аудио"""
        import torch
        
        if self.model is None:
            self._load_model()
        
        # Определение языка
        lang = self.language_map.get(self.target_language, self.language)
        
        # Использование примера голоса или встроенного спикера
        if speaker_wav and Path(speaker_wav).exists():
            # Клонирование голоса
            self.model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=lang,
                file_path=str(output_file)
            )
        else:
            # Использование встроенного спикера
            self.model.tts_to_file(
                text=text,
                language=lang,
                file_path=str(output_file)
            )
        
        return str(output_file)
    
    def synthesize(self) -> List[Dict]:
        """
        Использование локальной модели Coqui XTTS v2 для синтеза
        Returns:
            List[Dict]: [{"start_time": 0.0, "end_time": 1.5, "text": "...", "filename": "path/to/audio.wav"}]
        """
        logger.info("Запуск локального Coqui XTTS v2 синтеза")
        logger.warning(f"Целевой язык: {self.target_language}, XTTS язык: {self.language_map.get(self.target_language, self.language)}")
        
        # Если не указан образец голоса, но есть исходное видео - извлекаем голос из него
        if not self.speaker_wav and self.source_video:
            self.speaker_wav = self._extract_speaker_from_video(self.source_video)
        elif not self.speaker_wav:
            logger.warning("Образец голоса не указан, используется встроенный спикер XTTS")
        
        for i, sub in enumerate(self.subtitles):
            if not sub.get("text", "").strip():
                continue
            
            output_file = self.output_path / f"audio_{i:04d}.wav"
            
            try:
                # Синтез аудио для текущего сегмента
                audio_path = self._synthesize_segment(
                    sub["text"], 
                    output_file,
                    speaker_wav=self.speaker_wav
                )
                sub["filename"] = audio_path
                logger.info(f"Сгенерировано аудио для сегмента {i+1}/{len(self.subtitles)}: {audio_path}")
                
            except Exception as e:
                logger.error(f"Ошибка синтеза для сегмента {i}: {e}")
                raise
        
        return self.subtitles
    
    def merge_audio(self, output_path: str) -> str:
        """Объединение всех аудиофрагментов в один файл с правильными таймингами"""
        from pydub import AudioSegment
        import numpy as np
        
        logger.info("Начало объединения аудиофрагментов...")
        
        # Сортируем субтитры по времени начала
        sorted_subs = sorted(self.subtitles, key=lambda x: x["start_time"])
        
        # Определяем общую длительность видео по последнему таймингу
        if sorted_subs:
            total_duration_ms = int(sorted_subs[-1]["end_time"] * 1000)
        else:
            total_duration_ms = 0
        
        # Создаём пустой аудиофайл нужной длительности
        merged = AudioSegment.silent(duration=total_duration_ms)
        
        for i, sub in enumerate(sorted_subs):
            if "filename" not in sub or not Path(sub["filename"]).exists():
                logger.warning(f"Аудиофайл для сегмента {i} не найден, пропускаем")
                continue
            
            try:
                audio = AudioSegment.from_wav(sub["filename"])
                
                # Обрезаем тишину в начале и конце аудио (очистка артефактов)
                audio = self._trim_silence(audio, silence_threshold=-50, min_silence_duration=100)
                
                # Рассчитываем позицию наложения
                start_position_ms = int(sub["start_time"] * 1000)
                
                # Обрезаем аудио если оно выходит за рамки end_time
                max_duration_ms = int(sub["end_time"] * 1000) - start_position_ms
                if len(audio) > max_duration_ms:
                    audio = audio[:max_duration_ms]
                    logger.warning(f"Сегмент {i} обрезан по таймингу")
                
                # Накладываем аудио на нужную позицию
                merged = merged.overlay(audio, position=start_position_ms)
                
                logger.debug(f"Сегмент {i}: начало={start_position_ms}мс, длительность={len(audio)}мс")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке сегмента {i}: {e}")
                continue
        
        # Экспортируем результат
        output_file = Path(output_path)
        merged.export(str(output_file), format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "16000"])
        logger.info(f"Объединенное аудио сохранено: {output_file}, общая длительность: {len(merged)}мс")
        return str(output_file)
    
    def _trim_silence(self, audio: "AudioSegment", silence_threshold: float = -50, min_silence_duration: int = 100) -> "AudioSegment":
        """
        Обрезает тишину в начале и конце аудио
        
        Args:
            audio: AudioSegment для обработки
            silence_threshold: Порог тишины в dBFS (по умолчанию -50)
            min_silence_duration: Минимальная длительность тишины в мс для обрезки
        
        Returns:
            AudioSegment с обрезанной тишиной
        """
        from pydub.silence import detect_leading_silence, detect_trailing_silence
        
        # Обрезаем тишину в начале
        leading_silence = detect_leading_silence(audio, silence_threshold=silence_threshold)
        if leading_silence > min_silence_duration:
            audio = audio[leading_silence:]
        
        # Обрезаем тишину в конце
        trailing_silence = detect_trailing_silence(audio, silence_threshold=silence_threshold)
        if trailing_silence > min_silence_duration:
            audio = audio[:-trailing_silence]
        
        return audio
