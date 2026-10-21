"""Teacher-forced output-error evaluation: score a reference continuation (the BF16 dense model's greedy output,
from validate_generation.py) with prompt_logprobs under the current configuration.

    python scripts/teacher_forced_eval.py --model ... --prompts prompts.json --reference gen-dense.json \
        --output tfe-dense.json                         # reference scores
    SM120_SPARSE_KERNEL=sm120fp8 python scripts/teacher_forced_eval.py ... --quantization fp8 \
        --reference gen-dense.json --reference-scores tfe-dense.json --output tfe-c.json
Reported per prompt: mean NLL of the reference tokens, top-1 agreement, mean/max |delta logprob| vs the reference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import add_engine_args, load_prompts, make_llm, sparse_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_engine_args(ap)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True, help='validate_generation.py output of the BF16 dense run')
    ap.add_argument('--reference-scores', type=Path, default=None, help="this script's output for the reference configuration")
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kinds', default='docs_needle')
    ap.add_argument('--lengths', default='65536,131072,260096')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--continuation-tokens', type=int, default=256)
    args = ap.parse_args()
    llm = make_llm(args)
    from vllm import SamplingParams
    _, cfg = sparse_config()
    key_of = lambda r: (r['kind'], r['length'], r.get('index', 0))
    ref_rows = {key_of(r): r for r in json.loads(args.reference.read_text())['rows']}
    ref_scores = {key_of(r): r for r in json.loads(args.reference_scores.read_text())['rows']} if args.reference_scores else {}
    rows = []
    for row in [r for r in load_prompts(args.prompts, args.kinds, args.lengths) if r.get('index', 0) == args.index]:
        key = key_of(row)
        cont = list(ref_rows[key]['token_ids'])[:args.continuation_tokens]
        ids = list(row['prompt_token_ids']) + cont
        t0 = time.time()
        res = llm.generate([{'prompt_token_ids': ids}], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        elapsed = time.time() - t0
        plp = res[0].prompt_logprobs
        start = len(row['prompt_token_ids'])
        lps, top1 = [], []
        for i in range(start, len(ids)):
            lp = plp[i][ids[i]]
            lps.append(lp.logprob)
            top1.append(1.0 if lp.rank == 1 else 0.0)
        nll = -sum(lps) / len(lps)
        rec = {'kind': row['kind'], 'length': row['length'], 'index': row.get('index', 0), 'prompt_sha256': row['prompt_sha256'],
               'continuation_tokens': len(cont), 'mean_nll': nll, 'ppl': math.exp(nll), 'top1_agreement': sum(top1) / len(top1),
               'logprobs': lps, 'elapsed_s': elapsed}
        if key in ref_scores:
            rl = ref_scores[key]['logprobs'][:len(lps)]
            d = [abs(a - b) for a, b in zip(lps, rl)]
            rec.update(delta_logprob_mean=sum(d) / len(d), delta_logprob_max=max(d), delta_nll=nll + sum(rl) / len(rl))
        rows.append(rec)
        print(f"{row['kind']}-{row['length']}: NLL {nll:.4f} top1 {rec['top1_agreement']:.3f}"
              + (f" dNLL {rec['delta_nll']:+.4f} d|lp| mean {rec['delta_logprob_mean']:.4f} max {rec['delta_logprob_max']:.3f}" if key in ref_scores else '')
              + f" ({elapsed:.0f}s)", flush=True)
    args.output.write_text(json.dumps({'config': cfg, 'engine': {k: getattr(args, k) for k in ('model', 'quantization', 'kv_cache_dtype')}, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
