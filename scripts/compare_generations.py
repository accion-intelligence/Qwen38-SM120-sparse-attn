"""Compare greedy generations of a dense run against sparse runs (token agreement, needle hits)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', type=Path, required=True)
    ap.add_argument('--sparse', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    dense = json.loads(args.dense.read_text())
    drows = {(r['kind'], r['length']): r for r in dense['rows']}
    lines = ['| config | prompt | kept blocks | first token | common prefix | agree/512 | needle dense→sparse |',
             '|---|---|---:|---|---:|---:|---|']
    summary = {}
    for sp in args.sparse:
        data = json.loads(sp.read_text())
        cfg = data['config']
        name = f"tau={cfg.get('tau')}" if cfg.get('enabled') else 'dense-rerun'
        for r in data['rows']:
            d = drows[(r['kind'], r['length'])]
            cp = common_prefix(d['token_ids'], r['token_ids'])
            agree = sum(1 for x, y in zip(d['token_ids'], r['token_ids']) if x == y)
            kept = r['sparse'].get('kept_fraction') if r['sparse'] else None
            lines.append(f"| {name} | {r['kind']}-{r['length']} | {kept if kept is None else f'{kept:.3f}'} | "
                         f"{'same' if d['token_ids'][:1] == r['token_ids'][:1] else 'DIFF'} | {cp} | {agree} | "
                         f"{d['needle_hit']}→{r['needle_hit']} |")
            summary.setdefault(name, []).append({'prompt': f"{r['kind']}-{r['length']}", 'kept': kept,
                                                 'first_token_same': d['token_ids'][:1] == r['token_ids'][:1],
                                                 'common_prefix': cp, 'agree': agree,
                                                 'needle_dense': d['needle_hit'], 'needle_sparse': r['needle_hit']})
    args.output.write_text(json.dumps(summary, indent=1))
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
