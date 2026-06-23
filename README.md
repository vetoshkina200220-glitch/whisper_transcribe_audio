# Whisper Local Transcription

[English](README.md) | [Русский](README.ru.md)

A command-line tool for transcribing audio files locally using [OpenAI Whisper](https://github.com/openai/whisper). No internet required after model download — everything runs on your machine.

## Features

- Transcribe any audio format supported by ffmpeg (mp3, wav, m4a, mp4, webm, ogg, …)
- Translate audio to English (`--task translate`)
- Multiple output formats: plain text, JSON, SRT subtitles, WebVTT subtitles
- Full control over all Whisper hyperparameters via CLI flags
- Runs on CPU, NVIDIA GPU (CUDA), or Apple Silicon (MPS)

---

## Requirements

| Dependency | Version | Install |
|------------|---------|---------|
| Python | 3.8 – 3.11 | [python.org](https://www.python.org/) |
| ffmpeg | any | see install instructions below |
| pip packages | see below | `pip install -r requirements.txt` |

---

## Installation

```bash
# 1. Clone or download the project
git clone https://github.com/EgorKA027/transcribe_audio_whisper.git

# 2. Navigate to the project folder
cd transcribe_audio_whisper/

# 3. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 4. Install Python dependencies
pip install -r requirements.txt

# 4. Verify ffmpeg is available
ffmpeg -version
```

**Installing ffmpeg:**

- **macOS:** `brew install ffmpeg`
- **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
- **Linux (Fedora/RHEL):** `sudo dnf install ffmpeg`
- **Windows:** Download a build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html), extract the archive, and add the `bin/` folder to your `PATH` environment variable:
  1. Extract the archive to a folder, e.g. `C:\ffmpeg`
  2. Open **Start** → search for **"Edit the system environment variables"** → click **Environment Variables**
  3. Under **System variables**, select **Path** → click **Edit** → **New**
  4. Enter `C:\ffmpeg\bin` and click **OK** in all windows
  5. Restart any open terminals, then verify with `ffmpeg -version`

---

## Quick Start

```bash
# Basic transcription (auto-detect language, base model)
python transcribe.py audio.mp3

# Save result to a file
python transcribe.py audio.mp3 --output result

# Specify language and a larger model
python transcribe.py audio.mp3 --model small --language ru

# Generate SRT subtitles
python transcribe.py audio.mp3 --output subtitles --output-format srt

# Translate any language to English
python transcribe.py audio.mp3 --task translate --output translated
```

---

## Model Comparison

| Model | Parameters | VRAM | Relative Speed | Best For |
|-------|-----------|------|---------------|----------|
| `tiny` | 39 M | ~1 GB | ~32x | Quick tests, low-resource machines |
| `tiny.en` | 39 M | ~1 GB | ~32x | Same as tiny, English only — slightly more accurate |
| `base` | 74 M | ~1 GB | ~16x | Good balance for everyday use |
| `base.en` | 74 M | ~1 GB | ~16x | English-only, slightly better accuracy |
| `small` | 244 M | ~2 GB | ~6x | Better multilingual accuracy |
| `small.en` | 244 M | ~2 GB | ~6x | English: near-medium quality |
| `medium` | 769 M | ~5 GB | ~2x | High accuracy, all languages |
| `medium.en` | 769 M | ~5 GB | ~2x | Best English accuracy below large |
| `large` | 1550 M | ~10 GB | 1x | Maximum accuracy, multilingual |
| `large-v2` | 1550 M | ~10 GB | 1x | Improved large |
| `large-v3` | 1550 M | ~10 GB | 1x | Best overall accuracy |
| `turbo` | 798 M | ~6 GB | ~8x | Near-large accuracy at much higher speed |

**Recommendation:** Start with `base` for testing. Use `small` or `medium` for production. Use `large-v3` when accuracy is critical and you have enough RAM/VRAM.

---

## All CLI Arguments

### Input / Output

| Argument | Default | Description |
|----------|---------|-------------|
| `audio` | *(required)* | Path to the audio file |
| `--output PATH` | stdout | Output file path without extension |
| `--output-format` | `txt` | `txt` · `json` · `srt` · `vtt` |

### Model & Device

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `base` | Model size (see table above) |
| `--device` | `cpu` | `cpu` · `cuda` · `mps` (falls back to cpu if unavailable) |

### Language & Task

| Argument | Default | Description |
|----------|---------|-------------|
| `--language CODE` | auto | Language code: `en`, `ru`, `de`, `fr`, `ja`, … Setting this speeds up detection. |
| `--task` | `transcribe` | `transcribe` keeps original language; `translate` outputs English |

### Decoding Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature FLOAT` | `0.0` | Sampling temperature. 0 = greedy (deterministic). Higher values increase variety but may reduce accuracy. |
| `--beam-size INT` | `5` | Beam search width. Higher = better quality, slower. Set to 1 for greedy decoding. |
| `--best-of INT` | `5` | Number of candidates evaluated when `temperature > 0`. |

### Precision

| Argument | Description |
|----------|-------------|
| `--fp16` | Force half-precision (fp16). Faster on GPU. |
| `--no-fp16` | Force full-precision (fp32). Recommended for CPU. |

> By default, fp16 is used automatically on CUDA; fp32 is used on CPU and MPS.

### Quality Thresholds

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-speech-threshold FLOAT` | `0.6` | If the model's no-speech probability exceeds this, the segment is skipped. Lower = stricter silence detection. |
| `--compression-ratio-threshold FLOAT` | `2.4` | Segments with unusually high compression ratio are likely hallucinations and get discarded. |
| `--logprob-threshold FLOAT` | `-1.0` | Segments with average log-probability below this threshold are discarded. |

### Context & Prompting

| Argument | Default | Description |
|----------|---------|-------------|
| `--condition-on-previous-text` / `--no-condition-on-previous-text` | enabled | Feed previous segment output as context. Disable if you notice repetition loops. |
| `--initial-prompt TEXT` | none | A text hint to guide vocabulary and style. Example: `--initial-prompt "Медицинский разговор"` |

### Extras

| Argument | Default | Description |
|----------|---------|-------------|
| `--word-timestamps` | off | Add word-level timestamps to JSON output and subtitle formats. |
| `--verbose` | off | Print each segment as it is transcribed. When off, a tqdm progress bar is shown instead. |

---

## Progress Bar

By default, transcription shows a `tqdm` progress bar in the console:

```
  8%|████▌                                                   | 1344/16000 [00:03<00:35, 415.06frames/s]
```

It displays:
- Frames processed out of total
- Elapsed time and estimated time remaining
- Processing speed in frames per second

Use `--verbose` to switch from the progress bar to segment-by-segment text output instead.

---

## Logging

Every run writes log messages both to the console and to a date-stamped file inside `logs/`:

```
logs/
└── 2026-05-15.log
```

- The `logs/` directory is created automatically on first run.
- Each day gets its own file (`YYYY-MM-DD.log`). Multiple runs on the same day **append** to the existing file — nothing is overwritten.
- Console output shows `INFO`-level messages and above.
- The log file captures everything from `DEBUG` level up, with full timestamps.

Example log file content:
```
2026-05-15 12:00:01,234 [INFO] Loading model 'base' on cpu...
2026-05-15 12:00:05,678 [INFO] Transcribing 'audio.mp3'...
2026-05-15 12:00:12,910 [INFO] Saved to result.txt
```

---

## Output Format Examples

**txt** — plain transcription:
```
Hello, this is a sample transcription of the audio file.
```

**json** — full structured result:
```json
{
  "text": "Hello, this is a sample...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.5, "text": "Hello, this is a sample..."}
  ],
  "language": "en"
}
```

**srt** — SubRip subtitles:
```
1
00:00:00,000 --> 00:00:03,500
Hello, this is a sample transcription.
```

**vtt** — WebVTT subtitles:
```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, this is a sample transcription.
```

---

## Usage Examples

```bash
# Russian audio, medium model, save as SRT
python transcribe.py interview.m4a --model medium --language ru --output interview --output-format srt

# Translate French podcast to English text
python transcribe.py podcast.mp3 --task translate --output podcast_en

# High-accuracy transcription with word timestamps
python transcribe.py lecture.wav --model large-v3 --word-timestamps --output lecture --output-format json

# Fast transcription for testing (tiny model, greedy decoding)
python transcribe.py draft.mp3 --model tiny --beam-size 1

# Apple Silicon GPU acceleration
python transcribe.py audio.mp3 --model small --device mps
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Egor Kainov
