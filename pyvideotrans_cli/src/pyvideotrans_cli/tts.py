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
                
                # НЕ обрезаем тишину! Мы уже обрезали её при синтезе.
                # audio = self._trim_silence(audio, silence_threshold=-50, min_silence_duration=100)
                
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
        # Импорт удалён - функции реализованы вручную ниже
        
        # Обрезаем тишину в начале
        leading_silence = self._detect_leading_silence(audio, silence_threshold=silence_threshold)
        if leading_silence > min_silence_duration:
            audio = audio[leading_silence:]
        
        # Обрезаем тишину в конце
        trailing_silence = self._detect_trailing_silence(audio, silence_threshold=silence_threshold)
        if trailing_silence > min_silence_duration:
            audio = audio[:-trailing_silence]
        
        return audio
    
    def _detect_leading_silence(self, audio, silence_threshold=-50):
        """Обнаруживает тишину в начале аудио (в мс)"""
        silence_duration = 0
        for i in range(0, len(audio), 10):  # шаг 10 мс
            segment = audio[i:i+10]
            if segment.dBFS < silence_threshold:
                silence_duration += 10
            else:
                break
        return silence_duration
    
    def _detect_trailing_silence(self, audio, silence_threshold=-50):
        """Обнаруживает тишину в конце аудио (в мс)"""
        silence_duration = 0
        for i in range(len(audio) - 10, -1, -10):  # шаг 10 мс назад
            segment = audio[i:i+10]
            if segment.dBFS < silence_threshold:
                silence_duration += 10
            else:
                break
        return silence_duration


@dataclass
class CosyVoiceTTS:
    """CosyVoice 300M - локальная модель TTS от Alibaba с поддержкой клонирования голоса"""
    subtitles: List[Dict]
    target_language: str = "en"
    model_name: str = "iic/CosyVoice-300M"
    output_dir: str = "./output"
    device: str = "cuda"  # "cuda" или "cpu"
    speaker_wav: Optional[str] = None  # Путь к примеру голоса для клонирования
    language: str = "ru"  # Язык синтеза
    source_video: Optional[str] = None  # Исходное видео для извлечения голоса
    
    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.frontend = None
        
        # Маппинг кодов языков для CosyVoice
        self.language_map = {
            "en": "English",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "pl": "Polish",
            "tr": "Turkish",
            "ru": "Russian",
            "nl": "Dutch",
            "cs": "Czech",
            "ar": "Arabic",
            "hu": "Hungarian",
            "ko": "Korean",
            "ja": "Japanese",
            "hi": "Hindi",
        }
    
    def _extract_speaker_from_video(self, video_path: str) -> str:
        """Извлечение образца голоса из исходного видео (первые 10-15 секунд с речью)"""
        import subprocess
        
        speaker_file = self.output_path / "speaker_sample.wav"
        
        logger.info(f"Извлечение образца голоса из видео: {video_path}")
        
        # Извлекаем аудио для использования в качестве образца голоса
        # CosyVoice лучше работает с чистым голосом, поэтому берём 10-15 секунд
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-t", "15",  # Первые 15 секунд
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # CosyVoice использует 16000 Гц
            "-ac", "1",
            str(speaker_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error при извлечении голоса: {result.stderr}")
        
        logger.info(f"Образец голоса сохранён: {speaker_file}")
        return str(speaker_file)
    
    def _load_model(self):
        """Загрузка модели CosyVoice через funasr"""
        if self.model is not None:
            return
            
        try:
            from funasr import AutoModel
            import torch
            
            logger.info(f"Загрузка локальной модели CosyVoice: {self.model_name}")
            
            # Определение доступных устройств
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA не доступна, переключение на CPU")
                self.device = "cpu"
            
            # Загрузка модели CosyVoice через ModelScope
            self.model = AutoModel(
                model=self.model_name,
                device=self.device,
                disable_update=True  # Не проверять обновления
            )
            
            logger.info("Модель CosyVoice успешно загружена")
            
        except ImportError as e:
            logger.error(f"Необходимые библиотеки не установлены: {e}")
            raise RuntimeError("Установите зависимости: pip install funasr modelscope")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def _synthesize_segment(self, text: str, output_file: Path, speaker_wav: Optional[str] = None) -> str:
        """Синтез одного сегмента текста в аудио с использованием клонирования голоса"""
        import torch
        import soundfile as sf
        import librosa
        
        if self.model is None:
            self._load_model()
        
        # Определение языка
        lang_name = self.language_map.get(self.target_language, "Russian")
        
        # Загрузка референсного аудио для клонирования голоса
        prompt_speech_16k = None
        if speaker_wav and Path(speaker_wav).exists():
            try:
                # Загружаем аудио и ресемплим до 16kHz если нужно
                speech, sample_rate = librosa.load(speaker_wav, sr=16000)
                prompt_speech_16k = torch.from_numpy(speech).unsqueeze(0).float()
                logger.debug(f"Образец голоса загружен: {speaker_wav}, длительность: {len(speech)/16000:.2f}с")
            except Exception as e:
                logger.warning(f"Не удалось загрузить образец голоса: {e}, используется спикер по умолчанию")
        
        try:
            # Генерация аудио через CosyVoice
            # CosyVoice возвращает список с результатом
            result = self.model.generate(
                text=text,
                prompt_speech_16k=prompt_speech_16k,
                lang=lang_name,
                stream=False,
                speed=1.0,  # Нормальная скорость для сохранения окончаний
            )
            
            # Извлекаем аудио из результата
            # result обычно содержит [{'wav': tensor, 'sample_rate': int}]
            if isinstance(result, list) and len(result) > 0:
                audio_data = result[0].get('wav', None)
                sample_rate = result[0].get('sample_rate', 16000)
            elif isinstance(result, dict):
                audio_data = result.get('wav', None)
                sample_rate = result.get('sample_rate', 16000)
            else:
                raise RuntimeError(f"Неожиданный формат результата: {type(result)}")
            
            if audio_data is None:
                raise RuntimeError("Модель вернула пустой результат")
            
            # Конвертируем тензор в numpy
            if hasattr(audio_data, 'cpu'):
                audio_np = audio_data.cpu().numpy()
            else:
                audio_np = audio_data
            
            # Нормализация к диапазону [-1, 1]
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            
            max_val = np.max(np.abs(audio_np))
            if max_val > 0 and max_val > 1.0:
                audio_np = audio_np / max_val
            
            # Сохраняем аудиофайл
            sf.write(str(output_file), audio_np, samplerate=sample_rate)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Ошибка синтеза: {e}")
            raise
    
    def _resize_audio_to_duration(self, audio_path: Path, target_duration_ms: int) -> str:
        """Растягивает или сжимает аудио под нужный тайминг без изменения питча"""
        from pydub import AudioSegment
        import numpy as np
        
        audio = AudioSegment.from_wav(str(audio_path))
        current_duration_ms = len(audio)
        
        # Если аудио уже подходит по длительности (±100мс), не трогаем его
        if abs(current_duration_ms - target_duration_ms) < 100:
            return str(audio_path)
        
        # Вычисляем коэффициент масштабирования
        speed_factor = current_duration_ms / target_duration_ms
        
        # Используем питчинг для изменения скорости без изменения тона
        # Для CosyVoice мы просто растягиваем/сжимаем через изменение sample rate
        import soundfile as sf
        import librosa
        
        # Загружаем аудио
        data, sample_rate = sf.read(str(audio_path))
        
        # Ресемплим для изменения длительности
        target_length = int(len(data) / speed_factor)
        resampled_data = librosa.resample(
            data, 
            orig_sr=sample_rate, 
            target_sr=int(sample_rate * speed_factor)
        )
        
        # Обрезаем или дополняем до нужной длины
        if len(resampled_data) > target_length:
            resampled_data = resampled_data[:target_length]
        elif len(resampled_data) < target_length:
            # Дополняем тишиной если нужно
            resampled_data = np.pad(resampled_data, (0, target_length - len(resampled_data)))
        
        # Сохраняем с оригинальным sample rate
        output_path = audio_path.with_suffix('.resized.wav')
        sf.write(str(output_path), resampled_data, sample_rate)
        
        return str(output_path)
    
    def synthesize(self) -> List[Dict]:
        """
        Использование локальной модели CosyVoice для синтеза с клонированием голоса
        Returns:
            List[Dict]: [{"start_time": 0.0, "end_time": 1.5, "text": "...", "filename": "path/to/audio.wav"}]
        """
        logger.info("Запуск локального CosyVoice синтеза")
        logger.warning(f"Целевой язык: {self.target_language}, CosyVoice язык: {self.language_map.get(self.target_language, 'Russian')}")
        
        # Если не указан образец голоса, но есть исходное видео - извлекаем голос из него
        if not self.speaker_wav and self.source_video:
            self.speaker_wav = self._extract_speaker_from_video(self.source_video)
        elif not self.speaker_wav:
            logger.warning("Образец голоса не указан, используется спикер по умолчанию")
        
        for i, sub in enumerate(self.subtitles):
            if not sub.get("text", "").strip():
                continue
            
            output_file = self.output_path / f"audio_{i:04d}.wav"
            target_duration_ms = int((sub["end_time"] - sub["start_time"]) * 1000)
            
            try:
                # Синтез аудио для текущего сегмента
                audio_path = self._synthesize_segment(
                    sub["text"], 
                    output_file,
                    speaker_wav=self.speaker_wav
                )
                
                # Проверяем длительность и при необходимости растягиваем/сжимаем
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(audio_path)
                current_duration_ms = len(audio)
                
                # Если аудио слишком короткое или длинное - адаптируем
                if current_duration_ms < target_duration_ms * 0.8 or current_duration_ms > target_duration_ms * 1.1:
                    logger.info(f"Сегмент {i}: адаптация длительности {current_duration_ms}мс -> {target_duration_ms}мс")
                    audio_path = self._resize_audio_to_duration(Path(audio_path), target_duration_ms)
                
                sub["filename"] = audio_path
                logger.info(f"Сгенерировано аудио для сегмента {i+1}/{len(self.subtitles)}: {audio_path}")
                
            except Exception as e:
                logger.error(f"Ошибка синтеза для сегмента {i}: {e}")
                raise
        
        return self.subtitles
    
    def merge_audio(self, output_path: str) -> str:
        """Объединение всех аудиофрагментов в один файл с правильными таймингами"""
        from pydub import AudioSegment
        
        logger.info("Начало объединения аудиофрагментов...")
        
        # Сортируем субтитры по времени начала
        sorted_subs = sorted(self.subtitles, key=lambda x: x["start_time"])
        
        # Определяем общую длительность видео по последнему таймингу
        if sorted_subs:
            total_duration_ms = int(sorted_subs[-1]["end_time"] * 1000) + 100  # +100мс запас
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
                
                # Рассчитываем позицию наложения
                start_position_ms = int(sub["start_time"] * 1000)
                
                # Проверяем что аудио не выходит за рамки end_time
                max_duration_ms = int(sub["end_time"] * 1000) - start_position_ms
                if len(audio) > max_duration_ms:
                    # Мягкое затухание в конце если обрезаем
                    audio = audio[:max_duration_ms].fade_out(50)
                    logger.debug(f"Сегмент {i}: обрезан до {max_duration_ms}мс")
                
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
