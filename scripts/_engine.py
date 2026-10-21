"""Shared helpers for the offline evaluation scripts: build a vLLM LLM from CLI flags, read prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--model', required=True)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--max-model-len', type=int, default=262144)
    ap.add_argument('--max-num-batched-tokens', type=int, default=8192, help='prefill chunk; 12032 aligns the attention grid to whole waves on 188 SMs')
    ap.add_argument('--max-num-seqs', type=int, default=1)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--quantization', default=None, help="e.g. fp8 (W8A8 dynamic FP8 GEMM)")
    ap.add_argument('--kv-cache-dtype', default=None)
    ap.add_argument('--no-enforce-eager', action='store_true', help='allow CUDA graphs (the scripts default to eager)')


def make_llm(args):
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')   # in-process engine: sparse statistics are readable
    from vllm import LLM
    extra = {}
    if args.quantization:
        extra['quantization'] = args.quantization
    if args.kv_cache_dtype:
        extra['kv_cache_dtype'] = args.kv_cache_dtype
    return LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len, enforce_eager=not args.no_enforce_eager,
               gpu_memory_utilization=args.gpu_memory_utilization, max_num_batched_tokens=args.max_num_batched_tokens,
               max_num_seqs=args.max_num_seqs, enable_chunked_prefill=True, enable_prefix_caching=False, seed=20261015, **extra)


def sparse_config():
    try:
        from sm120_attn import sparse_prefill as ksp
        return ksp, ksp.config()
    except Exception:
        return None, {'enabled': False}


def load_prompts(path: Path, kinds: str, lengths: str):
    want = {int(x) for x in lengths.split(',')}
    return [r for r in json.loads(Path(path).read_text()) if r['kind'] in kinds.split(',') and r['length'] in want]
