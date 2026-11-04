"""
Backend — HuggingFace Model Architecture Analyzer

Uses lazy imports to avoid loading torch/transformers at import time.
"""

__version__ = "0.2.0"


def __getattr__(name):
    if name == "analyze_model":
        from .analyzer import analyze_model
        return analyze_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["analyze_model"]
