import argparse

import torch

from src.logger import logger


def resolve_device(requested: str) -> str:
    """Validate the requested compute device and fall back to CPU if unavailable.

    Checks hardware availability at runtime. If the requested device is not
    present, prints a warning to stderr and returns ``"cpu"`` instead of
    raising an exception, so transcription can always proceed.

    Parameters
    ----------
    requested : str
        Device name requested by the user. One of ``"cpu"``, ``"cuda"``,
        or ``"mps"``.

    Returns
    -------
    str
        The device that will actually be used: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA is not available on this machine. Falling back to CPU.")
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS (Apple Silicon GPU) is not available. Falling back to CPU.")
        return "cpu"

    return "cpu"


def resolve_fp16(args: argparse.Namespace, device: str) -> bool:
    """Determine whether to use half-precision (fp16) inference.

    Priority order:

    1. ``--no-fp16`` flag → always ``False``.
    2. ``--fp16`` flag → always ``True``.
    3. Auto-detect: ``True`` only when running on CUDA, ``False`` otherwise.

    fp16 is intentionally disabled by default on CPU and MPS because it offers
    no speed benefit on those devices and may reduce accuracy.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must expose ``args.fp16`` (bool) and
        ``args.no_fp16`` (bool).
    device : str
        The resolved device string (``"cpu"``, ``"cuda"``, or ``"mps"``),
        as returned by :func:`resolve_device`.

    Returns
    -------
    bool
        ``True`` if fp16 should be enabled, ``False`` otherwise.
    """
    if args.no_fp16:
        return False
    if args.fp16:
        return True
    return device == "cuda" and torch.cuda.is_available()
