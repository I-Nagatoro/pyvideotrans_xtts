"""
文本翻译模块 - 支持本地 LLM API 和 Transformers 本地翻译
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Translator:
    """翻译器 - 支持 LLM API 和本地 Transformers"""
    subtitles: List[Dict]
    target_language: str
    api_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    model_name: str = "local-model"
    max_token: int = 4096
    temperature: float = 0.2
    use_transformers: bool = False  # Флаг для использования локальной модели transformers
    
    def translate(self) -> List[Dict]:
        """
        执行翻译，返回翻译后的字幕列表
        Returns:
            List[Dict]: [{"start_time": 0.0, "end_time": 1.5, "text": "翻译后的文本"}]
        """
        if self.use_transformers:
            return self._translate_with_transformers()
        
        return self._translate_with_llm_api()
    
    def _translate_with_llm_api(self) -> List[Dict]:
        """Перевод через LLM API (OpenAI-совместимый)"""
        try:
            from openai import OpenAI
            import httpx
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
                http_client=httpx.Client()
            )
            
            # 批量翻译 (每批最多 10 条)
            batch_size = 10
            translated_subtitles = []
            
            for i in range(0, len(self.subtitles), batch_size):
                batch = self.subtitles[i:i + batch_size]
                batch_texts = [sub["text"] for sub in batch]
                
                prompt = self._build_prompt(batch_texts)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translation engine."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_token,
                    temperature=self.temperature
                )
                
                translated_text = response.choices[0].message.content.strip()
                
                # 解析翻译结果
                translated_lines = self._parse_translation(translated_text, len(batch_texts))
                
                for j, sub in enumerate(batch):
                    translated_subtitles.append({
                        "start_time": sub["start_time"],
                        "end_time": sub["end_time"],
                        "text": translated_lines[j] if j < len(translated_lines) else sub["text"]
                    })
                
                logger.info(f"Translated batch {i//batch_size + 1}")
            
            return translated_subtitles
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def _translate_with_transformers(self) -> List[Dict]:
        """Перевод через локальную модель Transformers (NLLB)"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            # Маппинг языковых кодов для NLLB
            lang_map = {
                'ru': 'rus_Cyrl',
                'en': 'eng_Latn',
                'zh': 'zho_Hans',
                'ja': 'jpn_Jpan',
                'ko': 'kor_Hang',
                'de': 'deu_Latn',
                'fr': 'fra_Latn',
                'es': 'spa_Latn',
                'it': 'ita_Latn',
                'pt': 'por_Latn',
            }
            
            target_lang_code = lang_map.get(self.target_language.lower(), f'{self.target_language}_Latn')
            
            logger.info(f"Loading NLLB model for translation to {self.target_language}...")
            
            # Загружаем модель и токенизатор
            model_name = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info(f"Using device: {device}")
            
            # Получаем ID целевого языка (совместимо с новыми версиями transformers)
            forced_bos_token_id = None
            if hasattr(tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = tokenizer.lang_code_to_id[target_lang_code]
            elif hasattr(tokenizer, 'additional_special_tokens_ids'):
                # Для новых версий transformers
                token_index = tokenizer.additional_special_tokens.index(target_lang_code)
                forced_bos_token_id = tokenizer.additional_special_tokens_ids[token_index]
            else:
                # Фоллбэк: ищем токен вручную
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang_code)
            
            translated_subtitles = []
            batch_size = 5  # Меньший размер батча для экономии памяти
            
            for i in range(0, len(self.subtitles), batch_size):
                batch = self.subtitles[i:i + batch_size]
                batch_texts = [sub["text"] for sub in batch]
                
                # Токенизация
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)
                
                # Генерация перевода
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512
                    )
                
                # Декодирование
                translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for j, sub in enumerate(batch):
                    translated_subtitles.append({
                        "start_time": sub["start_time"],
                        "end_time": sub["end_time"],
                        "text": translated_texts[j] if j < len(translated_texts) else sub["text"]
                    })
                
                logger.info(f"Translated batch {i//batch_size + 1} with Transformers")
            
            return translated_subtitles
            
        except Exception as e:
            logger.error(f"Transformers translation failed: {e}")
            raise
    
    def _build_prompt(self, texts: List[str]) -> str:
        """构建翻译提示词"""
        input_text = "\n".join(texts)
        return f"""Translate the following text to {self.target_language}. 
Only output the translation, one line per input line. Do not include any explanations.

<input>
{input_text}
</input>

Translation:"""
    
    def _parse_translation(self, text: str, expected_lines: int) -> List[str]:
        """解析翻译结果"""
        # 尝试提取 <TRANSLATE_TEXT> 标签内容
        match = re.search(r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 如果行数不匹配，尝试其他解析方式
        if len(lines) != expected_lines:
            # 可能返回的是单行，用标点分割
            if len(lines) == 1 and expected_lines > 1:
                import re
                lines = re.split(r'[.!?。！？;；]', lines[0])
                lines = [l.strip() for l in lines if l.strip()]
        
        return lines[:expected_lines]
    
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
        logger.info(f"Translated SRT saved to: {output_file}")
        return str(output_file)
