"""Process-wide compatibility patches loaded automatically by Python's site module."""

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda self: list(getattr(self, "all_special_tokens", []))
        )
except Exception:
    pass
