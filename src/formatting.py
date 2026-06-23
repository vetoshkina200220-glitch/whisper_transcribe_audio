import json


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    """Convert a duration in seconds to a subtitle timestamp string.

    Produces ``HH:MM:SS,mmm`` for SRT or ``HH:MM:SS.mmm`` for VTT,
    matching the format expected by subtitle players and editors.

    Parameters
    ----------
    seconds : float
        Duration in seconds (non-negative).
    vtt : bool, default False
        When ``True``, use a period as the decimal separator (WebVTT format).
        When ``False``, use a comma (SubRip / SRT format).

    Returns
    -------
    str
        Timestamp string in the form ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm``.

    Examples
    --------
    >>> format_timestamp(0.0)
    '00:00:00,000'
    >>> format_timestamp(3661.5)
    '01:01:01,500'
    >>> format_timestamp(3661.5, vtt=True)
    '01:01:01.500'
    """
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    sep = "." if vtt else ","

    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(result: dict) -> str:
    """Render a Whisper transcription result as SubRip (SRT) subtitle text.

    Each segment from ``result["segments"]`` becomes one numbered SRT block
    with start/end timestamps and the segment text.

    Parameters
    ----------
    result : dict
        Whisper transcription result as returned by ``model.transcribe()``.
        Must contain a ``"segments"`` key with a list of dicts, each having
        ``"start"`` (float), ``"end"`` (float), and ``"text"`` (str) fields.

    Returns
    -------
    str
        Full SRT-formatted string ready to be written to a ``.srt`` file.

    Examples
    --------
    >>> result = {
    ...     "segments": [
    ...         {"start": 0.0, "end": 2.5, "text": " Hello world."},
    ...         {"start": 2.5, "end": 5.0, "text": " How are you?"},
    ...     ]
    ... }
    >>> print(to_srt(result))
    1
    00:00:00,000 --> 00:00:02,500
    Hello world.
    <BLANKLINE>
    2
    00:00:02,500 --> 00:00:05,000
    How are you?
    """
    lines = []
    for i, seg in enumerate(result["segments"], start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


def to_vtt(result: dict) -> str:
    """Render a Whisper transcription result as WebVTT subtitle text.

    Produces a valid WebVTT file starting with the required ``WEBVTT`` header,
    followed by one cue block per segment.

    Parameters
    ----------
    result : dict
        Whisper transcription result as returned by ``model.transcribe()``.
        Must contain a ``"segments"`` key with a list of dicts, each having
        ``"start"`` (float), ``"end"`` (float), and ``"text"`` (str) fields.

    Returns
    -------
    str
        Full WebVTT-formatted string ready to be written to a ``.vtt`` file.

    Examples
    --------
    >>> result = {
    ...     "segments": [
    ...         {"start": 0.0, "end": 2.5, "text": " Hello world."},
    ...     ]
    ... }
    >>> print(to_vtt(result))
    WEBVTT
    <BLANKLINE>
    00:00:00.000 --> 00:00:02.500
    Hello world.
    """
    lines = ["WEBVTT\n"]
    for seg in result["segments"]:
        start = format_timestamp(seg["start"], vtt=True)
        end = format_timestamp(seg["end"], vtt=True)
        lines.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


def render_output(result: dict, fmt: str) -> str:
    """Serialize a Whisper transcription result to the requested format.

    Dispatches to the appropriate formatter based on ``fmt`` and returns
    the result as a string ready for printing or writing to a file.

    Parameters
    ----------
    result : dict
        Whisper transcription result as returned by ``model.transcribe()``.
        Expected keys: ``"text"`` (str), ``"segments"`` (list), ``"language"`` (str).
    fmt : str
        Output format. One of:

        - ``"txt"``  — plain transcription text, whitespace stripped.
        - ``"json"`` — full result dict serialized to indented JSON.
        - ``"srt"``  — SubRip subtitle format with timestamps.
        - ``"vtt"``  — WebVTT subtitle format with timestamps.

    Returns
    -------
    str
        Formatted transcription string.

    Raises
    ------
    ValueError
        If ``fmt`` is not one of the supported format strings.
    """
    if fmt == "txt":
        return result["text"].strip()
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    if fmt == "srt":
        return to_srt(result)
    if fmt == "vtt":
        return to_vtt(result)
    raise ValueError(f"Unknown format: {fmt!r}. Expected one of: txt, json, srt, vtt.")
