# CLAUDE.md — Whisper Transcription Project

## Environment Rules

- **Always** run Python via `.venv/bin/python`, never via system `python3`.
- **Always** install packages via `.venv/bin/pip install`, never globally.
- `ffmpeg` is a system dependency (installed via Homebrew). Verify presence with `which ffmpeg`. Do not attempt to install it via pip.
- Activate the environment in shell with `source .venv/bin/activate` before running any command.

## Code Rules

- **All imports must be at the top of the file.** Never place `import` statements inside functions, conditionals, or any other block. This applies to every file in the project without exception.
- Follow this rule even for heavy libraries like `torch` or `whisper` — import them at the module level.

## Project Structure

```
whisper/
├── .venv/              # Virtual environment (not committed)
├── transcribe.py       # Entry point: imports + main() only
├── src/
│   ├── __init__.py
│   ├── cli.py          # CLI argument parsing (parse_args)
│   ├── device.py       # Device and fp16 resolution (resolve_device, resolve_fp16)
│   └── formatting.py   # Output formatting (render_output, to_srt, to_vtt, …)
├── requirements.txt    # Python dependencies
├── CLAUDE.md           # This file
├── README.md           # User-facing documentation (English)
├── README.ru.md        # User-facing documentation (Russian)
└── .gitignore          # Git exclusions
```

## Running the Script

```bash
source .venv/bin/activate
python transcribe.py <audio_file> [options]
```

Or without activating:
```bash
.venv/bin/python transcribe.py <audio_file> [options]
```

## How to Work on This Project

Follow the plan below **strictly one step at a time**. Mark each step as `[x]` when done before moving to the next. Do not skip or combine steps.

### Implementation Plan & Progress

- [x] Step 1 — Create `.venv` virtual environment
- [x] Step 2 — Create `requirements.txt` and install dependencies
- [x] Step 3 — Create `transcribe.py` CLI script
- [x] Step 4 — Create `CLAUDE.md`
- [x] Step 5 — Create `README.md`
- [x] Step 6 — Create `.gitignore`
- [x] Step 7 — Refactor: split code into `src/cli.py`, `src/device.py`, `src/formatting.py`
- [ ] Step 8 — Verify: run `--help`, then test with a real audio file

## Key Design Decisions

- All Whisper hyperparameters are exposed as CLI arguments (`--temperature`, `--beam-size`, etc.) so the script requires no code edits for tuning.
- `--device` defaults to `cpu`; if `cuda` or `mps` is requested but unavailable, a warning is printed and CPU is used.
- fp16 is auto-enabled only on CUDA. Pass `--fp16` to force it, `--no-fp16` to disable.
- Output goes to stdout by default; use `--output <path>` to save to a file.
- Supported output formats: `txt`, `json`, `srt`, `vtt`.
