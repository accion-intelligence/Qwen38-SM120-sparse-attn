"""A/B/C output-error table: teacher-forced NLL / top-1 agreement (tfe-*.json) plus needle hits (gen-*.json)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation', type=Path, required=True, help='validation directory with tfe-*.json and gen-*.json')
    ap.add_argument('--variants', nargs='+', required=True, help='label=tfe_name[:gen_name] (first is the reference)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = []
    ref = None
    for item in args.variants:
        label, names = item.split('=', 1)
        tfe_name, _, gen_name = names.partition(':')
        tfe = json.loads((args.validation / f'tfe-{tfe_name}.json').read_text())
        gen = json.loads((args.validation / f'gen-{gen_name}.json').read_text()) if gen_name and (args.validation / f'gen-{gen_name}.json').exists() else None
        per_len = {r['length']: r for r in tfe['rows']}
        needles = {r['length']: r['needle_hit'] for r in gen['rows']} if gen else {}
        rows.append((label, per_len, needles))
        if ref is None:
            ref = per_len
    lengths = sorted(ref)
    out = ['| variant | ' + ' | '.join(f'{L} tok NLL (Δ) / top-1 / needle' for L in lengths) + ' | mean Δ NLL | mean top-1 |', '|---|' + '---|' * len(lengths) + '---:|---:|']
    summary = {}
    for label, per_len, needles in rows:
        cells, dn, t1 = [], [], []
        for L in lengths:
            r = per_len.get(L)
            if r is None:
                cells.append('-'); continue
            d = r['mean_nll'] - ref[L]['mean_nll']
            dn.append(d); t1.append(r['top1_agreement'])
            nd = needles.get(L)
            cells.append(f"{r['mean_nll']:.4f} ({d:+.4f}) / {100*r['top1_agreement']:.1f}% / {'✓' if nd else ('✗' if nd is False else '?')}")
        summary[label] = {'delta_nll': {L: per_len[L]['mean_nll'] - ref[L]['mean_nll'] for L in lengths if L in per_len},
                          'top1': {L: per_len[L]['top1_agreement'] for L in lengths if L in per_len}, 'needle': needles}
        out.append(f'| {label} | ' + ' | '.join(cells) + f" | {sum(dn)/len(dn):+.4f} | {100*sum(t1)/len(t1):.1f}% |")
    text = '\n'.join(out) + '\n'
    args.output.write_text(json.dumps(summary, indent=1)); args.output.with_suffix('.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
