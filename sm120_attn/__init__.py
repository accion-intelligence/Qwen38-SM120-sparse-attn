"""Dynamic block-sparse FP8/BF16 prefill attention for vLLM on SM120 (RTX PRO 6000 Blackwell)."""
__version__ = '0.1.0'
from .kernels import attention_sm120, attention_sm120_fp8, attention_sm120_fp8_tma  # noqa: F401
