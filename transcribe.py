#!/usr/bin/env python3
"""CLI entry point for local audio transcription using OpenAI Whisper."""

import os
import warnings

import whisper

from src.cli import parse_args
from src.device import resolve_device, resolve_fp16
from src.formatting import render_output
from src.logger import logger


def main() -> None:
    """Run the transcription pipeline from CLI arguments to output."""
    args = parse_args()

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    device = resolve_device(args.device)
    fp16 = resolve_fp16(args, device)

    logger.info("Loading model '%s' on %s...", args.model, device)
    model = whisper.load_model(args.model, device=device)

    transcribe_kwargs = dict(
        language=args.language,
        task=args.task,
        temperature=args.temperature,
        beam_size=args.beam_size,
        best_of=args.best_of,
        fp16=fp16,
        word_timestamps=args.word_timestamps,
        verbose=True if args.verbose else False,
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        logprob_threshold=args.logprob_threshold,
        condition_on_previous_text=args.condition_on_previous_text,
        initial_prompt=args.initial_prompt,
    )

    logger.info("Transcribing '%s'...", args.audio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.transcribe(args.audio, **transcribe_kwargs)

    output_text = render_output(result, args.output_format)

    if args.output:
        out_path = f"{args.output}.{args.output_format}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        logger.info("Saved to %s", out_path)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
