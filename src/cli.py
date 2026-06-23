import argparse


def parse_args() -> argparse.Namespace:
    """Build the argument parser and parse command-line arguments.

    Defines all CLI arguments for the Whisper transcription tool, covering
    model selection, device, language, output format, and every transcription
    hyperparameter exposed by ``model.transcribe()``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following attributes:

        audio : str
            Path to the input audio file.
        model : str
            Whisper model name (e.g. ``"base"``, ``"large-v3"``).
        device : str
            Compute device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
        language : str or None
            BCP-47 language code, or ``None`` for auto-detection.
        task : str
            ``"transcribe"`` or ``"translate"``.
        output : str or None
            Output file path without extension, or ``None`` to print to stdout.
        output_format : str
            One of ``"txt"``, ``"json"``, ``"srt"``, ``"vtt"``.
        temperature : float
            Sampling temperature for decoding.
        beam_size : int
            Beam search width.
        best_of : int
            Number of candidates when ``temperature > 0``.
        fp16 : bool
            Force fp16 inference.
        no_fp16 : bool
            Force fp32 inference.
        no_speech_threshold : float
            Silence detection probability threshold.
        compression_ratio_threshold : float
            Hallucination filter threshold.
        logprob_threshold : float
            Minimum acceptable average log-probability per segment.
        condition_on_previous_text : bool
            Whether to use previous segment as context.
        initial_prompt : str or None
            Optional prompt to guide style and vocabulary.
        word_timestamps : bool
            Whether to include word-level timestamps.
        verbose : bool
            Whether to print segment-by-segment progress.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py audio.mp3
  python transcribe.py audio.mp3 --model small --language ru
  python transcribe.py audio.mp3 --task translate --output result --output-format txt
  python transcribe.py audio.mp3 --model large-v3 --word-timestamps --output result --output-format json
  python transcribe.py audio.mp3 --model medium --output subtitles --output-format srt
        """,
    )

    parser.add_argument("audio", help="Path to the audio file to transcribe.")

    # Model & device
    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large", "large-v2", "large-v3",
            "turbo",
        ],
        help="Whisper model to use (default: base).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on: cpu, cuda (NVIDIA GPU), or mps (Apple Silicon). "
             "Falls back to cpu if the chosen device is unavailable (default: cpu).",
    )

    # Task & language
    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'en', 'ru', 'de'. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' keeps the original language; 'translate' converts to English (default: transcribe).",
    )

    # Output
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path without extension. If omitted, result is printed to stdout.",
    )
    parser.add_argument(
        "--output-format",
        default="txt",
        choices=["txt", "json", "srt", "vtt"],
        help="Output format: txt (plain text), json (full result), srt (SubRip subtitles), "
             "vtt (WebVTT subtitles). Default: txt.",
    )

    # Sampling / decoding hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy decoding, higher = more random). Default: 0.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width. Larger values improve quality but are slower. Default: 5.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Number of candidate sequences evaluated when temperature > 0. Default: 5.",
    )

    # Precision
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Force half-precision (fp16) inference. Faster on GPU, not recommended for CPU.",
    )
    fp16_group.add_argument(
        "--no-fp16",
        action="store_true",
        default=False,
        help="Force full-precision (fp32) inference.",
    )

    # Quality / filtering thresholds
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Probability threshold to consider a segment as silence. Default: 0.6.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=2.4,
        help="Segments with compression ratio above this are treated as failed transcription. Default: 2.4.",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-1.0,
        help="Average log-probability threshold; segments below this are discarded. Default: -1.0.",
    )

    # Context
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Feed previous output as context for the next window (default: enabled).",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional text prompt to guide transcription style or vocabulary.",
    )

    # Extra output options
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        default=False,
        help="Include word-level timestamps in the output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print transcription progress segment by segment.",
    )

    args = parser.parse_args()

    return args
