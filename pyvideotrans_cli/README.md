# pyvideotrans-cli

精简版视频翻译 CLI 工具，基于 [pyvideotrans](https://github.com/jianchang512/pyvideotrans) 项目裁剪而成。

## 功能特点

✅ **保留的核心功能:**
- 🎵 视频→音频提取 (ffmpeg)
- 🎤 语音识别 (Whisper - faster-whisper / openai-whisper)
- 🌐 文本翻译 (本地 LLM API - OpenAI 兼容接口)
- 🔊 TTS 配音 (Qwen TTS API / Local)
- 🎬 音视频合并 (ffmpeg)

❌ **已移除的功能:**
- GUI 界面
- 30+ 种其他 TTS 渠道 (仅保留 Qwen TTS)
- 说话人分离 (diarization)
- 声音克隆
- 复杂的音频对齐逻辑
- 批量处理
- 其他翻译渠道 (仅保留本地 LLM)

## 安装

### 基础安装
```bash
cd pyvideotrans_cli
pip install -e .
```

### 带 CUDA 支持
```bash
pip install -e ".[cuda]"
```

### 开发环境
```bash
pip install -e ".[dev]"
```

## 依赖要求

- Python >= 3.10
- ffmpeg (必须预先安装)
- NVIDIA GPU + CUDA (可选，用于加速)

## 使用方法

### 完整流程：视频翻译
```bash
pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --qwen-api-key YOUR_QWEN_API_KEY \
    --output-dir ./output
```

### 分步骤执行

#### 1. 仅转录 (语音→字幕)
```bash
pyvideotrans-cli -i video.mp4 \
    --mode transcribe \
    --whisper-model base \
    --source-lang auto \
    --output-dir ./output
```

输出：`./output/source.srt`

#### 2. 仅翻译 (字幕→翻译字幕)
```bash
pyvideotrans-cli -i video.mp4 \
    --mode translate \
    --target-lang en \
    --llm-api http://localhost:1234/v1 \
    --llm-key not-needed \
    --llm-model qwen-7b \
    --output-dir ./output
```

输出：`./output/en.srt`

#### 3. 仅 TTS (翻译字幕→配音音频)
```bash
pyvideotrans-cli -i video.mp4 \
    --mode tts \
    --target-lang en \
    --qwen-api-key YOUR_KEY \
    --qwen-voice Cherry \
    --output-dir ./output
```

输出：`./output/dubbed.wav`

#### 4. 仅合并 (视频 + 配音→最终视频)
```bash
pyvideotrans-cli -i video.mp4 \
    --mode merge \
    --target-lang en \
    --add-subtitles \
    --hardsub \
    --output-dir ./output
```

输出：`./output/video_dubbed.mp4`

## 参数说明

### 核心参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入视频文件路径 (必选) | - |
| `-o, --output-dir` | 输出目录 | `./output` |
| `--mode` | 运行模式 | `full` |

### 转录参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--whisper-model` | Whisper 模型大小 | `tiny` |
| `--source-lang` | 源语言代码 | `auto` |
| `--cuda` | 使用 CUDA 加速 | `False` |

### 翻译参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--target-lang` | 目标语言代码 | - |
| `--transformers` | Использовать локальную модель NLLB вместо LLM API | `False` |
| `--llm-api` | LLM API 地址 | `http://localhost:1234/v1` |
| `--llm-key` | LLM API Key | `not-needed` |
| `--llm-model` | LLM 模型名称 | `local-model` |
| `--llm-provider` | LLM 提供商类型 | `openai` |

### TTS 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--qwen-api-key` | Qwen TTS API Key | - |
| `--qwen-model` | Qwen TTS 模型 | `qwen3-tts-flash` |
| `--qwen-voice` | Qwen TTS 音色 | `Cherry` |
| `--qwen-local` | 使用本地 Qwen TTS | `False` |
| `--qwen-local-model` | 本地 Qwen 模型大小 | `1.7B` |

### 合并参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--keep-original-audio` | 保留原始音频 (混合) | `False` |
| `--add-subtitles` | 添加字幕到视频 | `False` |
| `--hardsub` | 硬字幕 (烧录) | `False` |

## Pipeline 流程

```
┌─────────────┐
│  Input Video │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Extract Audio│ (ffmpeg)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Transcribe │ (Whisper)
└──────┬──────┘
       │ source.srt
       ▼
┌─────────────┐
│  Translate  │ (Local LLM)
└──────┬──────┘
       │ {lang}.srt
       ▼
┌─────────────┐
│     TTS     │ (Qwen TTS)
└──────┬──────┘
       │ dubbed.wav
       ▼
┌─────────────┐
│    Merge    │ (ffmpeg)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Output Video│
└─────────────┘
```

## 本地 LLM 配置示例

### 使用 LM Studio
```bash
# 启动 LM Studio 并加载模型
# API 地址：http://localhost:1234/v1

pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --llm-api http://localhost:1234/v1 \
    --llm-model local-model
```

### 使用 Ollama
```bash
# 启动 Ollama
ollama run qwen:7b

pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --llm-api http://localhost:11434/v1 \
    --llm-model qwen:7b
```

### Локальный перевод через Transformers (NLLB)

Для перевода без необходимости в API можно использовать модель NLLB от Meta:

```bash
pyvideotrans-cli -i video.mp4 \
    --target-lang ru \
    --transformers
```

Этот режим использует модель `facebook/nllb-200-distilled-600M`, которая поддерживает более 200 языков.
Модель будет автоматически загружена при первом запуске (требуется ~2GB места на диске).

#### Преимущества локального перевода:
- ✅ Не требуется API ключ
- ✅ Полностью офлайн работа
- ✅ Поддержка 200+ языков
- ✅ Бесплатно

#### Недостатки:
- ⚠️ Медленнее чем LLM API
- ⚠️ Требует больше памяти (минимум 4GB RAM)
- ⚠️ Качество перевода может быть ниже чем у больших LLM

#### Поддерживаемые языковые коды:
`ru`, `en`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `it`, `pt` и многие другие.

### 使用 vLLM
```bash
python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen-7B-Chat \
    --port 8000

pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --llm-api http://localhost:8000/v1 \
    --llm-model Qwen/Qwen-7B-Chat
```

## Qwen TTS 配置

### API 模式

#### 获取 API Key
1. 访问 https://dashscope.console.aliyun.com/
2. 注册/登录阿里云账号
3. 创建 API Key

#### 可用音色
- Cherry (女声，推荐)
- Emily (女声)
- Serena (女声)
- Ethan (男声)
- Jack (男声)

#### 使用示例
```bash
pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --qwen-api-key YOUR_API_KEY \
    --qwen-voice Cherry
```

### Локальная версия (через transformers)

#### Требования
- GPU с минимум 8GB VRAM (для модели 0.7B)
- Установленные зависимости: `pip install transformers accelerate torch torchaudio soundfile`

#### Использование
```bash
pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --qwen-local \
    --qwen-local-model "Qwen/Qwen3-TTS-0.7B" \
    --cuda
```

#### Поддерживаемые модели
- `Qwen/Qwen3-TTS-0.7B` (рекомендуется)
- `Qwen/Qwen3-TTS-1.5B` (требует больше VRAM)

Модель будет автоматически загружена из HuggingFace при первом запуске.

#### На CPU (медленнее)
```bash
pyvideotrans-cli -i video.mp4 \
    --target-lang en \
    --qwen-local \
    --qwen-local-model "Qwen/Qwen3-TTS-0.7B"
```

## TODO

- [x] Реализация локального Qwen TTS через transformers
- [x] Локальный перевод через Transformers (NLLB)
- [ ] Добавить оптимизацию длительности аудио (подгонка под тайминги)
- [ ] Добавить отображение прогресса (progress bar)
- [ ] Поддержка пакетной обработки файлов
- [ ] Улучшить обработку ошибок при загрузке моделей

## 许可证

MIT License (继承自原项目)

## 致谢

本项目基于 [pyvideotrans](https://github.com/jianchang512/pyvideotrans) 裁剪开发，感谢原作者的贡献。
