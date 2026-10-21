"""Greedy generations with the sparse hook active: needle retrieval, texts, block statistics.

    SM120_SPARSE_KERNEL=sm120fp8 python scripts/validate_generation.py --model ... --prompts prompts.json \
        --output gen-fp8.json --lengths 65536,131072,260096 --max-tokens 384 [--quantization fp8]
    SM120_SPARSE_ATTN=0 ... --output gen-dense.json      # the dense reference
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import add_engine_args, load_prompts, make_llm, sparse_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_engine_args(ap)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kinds', default='docs_needle')
    ap.add_argument('--lengths', default='65536,131072,260096')
    ap.add_argument('--max-tokens', type=int, default=384)
    ap.add_argument('--index', type=int, default=0, help='which of the per-length prompts to use (single-request mode)')
    ap.add_argument('--batched', action='store_true', help='submit all selected prompts together (mixed batches); needs --max-num-seqs > 1')
    args = ap.parse_args()
    llm = make_llm(args)
    from vllm import SamplingParams
    ksp, cfg = sparse_config()
    prompts = load_prompts(args.prompts, args.kinds, args.lengths)
    if not args.batched:
        prompts = [r for r in prompts if r.get('index', 0) == args.index]
    rows = []

    def reset_stats():
        if ksp is not None:
            for st in ksp._STATE.values():
                st.stats.update(calls=0, blocks_total=0, blocks_kept=0)
                st._kept_acc = None

    def stats():
        if ksp is None:
            return {}
        out = {}
        for st in ksp._STATE.values():
            kept = st.kept_fraction() if st.kernel in ('sm120', 'sm120fp8') else None
            out = {'calls': st.stats['calls'], 'blocks_total': st.stats['blocks_total'], 'kept_fraction': kept}
        return out

    groups = [prompts] if args.batched else [[r] for r in prompts]
    for group in groups:
        reset_stats()
        t0 = time.time()
        res = llm.generate([{'prompt_token_ids': r['prompt_token_ids']} for r in group],
                           SamplingParams(temperature=0.0, max_tokens=args.max_tokens), use_tqdm=False)
        elapsed = time.time() - t0
        for row, r in zip(group, res):
            out = r.outputs[0]
            code = row.get('needle_code', '')
            rec = {'kind': row['kind'], 'length': row['length'], 'index': row.get('index', 0), 'prompt_sha256': row['prompt_sha256'],
                   'token_ids': list(out.token_ids), 'text': out.text, 'needle_hit': bool(code) and code in out.text,
                   'elapsed_s': elapsed, 'sparse': stats() if not args.batched else {}}
            rows.append(rec)
            print(f"{row['kind']}-{row['length']}#{row.get('index', 0)}: {len(out.token_ids)} tokens in {elapsed:.1f}s needle={rec['needle_hit']} sparse={rec['sparse']} | {out.text[:120]!r}", flush=True)
    args.output.write_text(json.dumps({'config': cfg, 'engine': {k: getattr(args, k) for k in ('model', 'quantization', 'kv_cache_dtype', 'max_num_batched_tokens')}, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
