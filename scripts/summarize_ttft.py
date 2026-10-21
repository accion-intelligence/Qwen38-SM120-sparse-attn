"""Summarize long-request TTFT runs (stock / fused / sparse) from run_long_service results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load(path: Path):
    r = json.loads((path / 'results.json').read_text())
    cases = {}
    for u in r['uniform']:
        if u['warmup']:
            continue
        cases.setdefault(u['case'], []).append((u['metrics']['ttft_mean_ms'], u['validation']['passed']))
    return r['status'], cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='name=path pairs, first is the reference')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    runs = []
    for item in args.runs:
        name, path = item.split('=', 1)
        status, cases = load(Path(path))
        runs.append((name, status, cases))
    ref_name, _, ref = runs[0]
    case_ids = sorted(ref, key=lambda c: (int(c.split('-')[0][2:]), int(c.split('-c')[1])))
    header = '| case | ' + ' | '.join(f'{n} ms' for n, _, _ in runs) + ' | ' + ' | '.join(f'{n} vs {ref_name}' for n, _, _ in runs[1:]) + ' | checks |'
    lines = [header, '|---|' + '---:|' * (len(runs) + len(runs) - 1) + '---|']
    summary = {}
    for c in case_ids:
        vals = [mean(v for v, _ in cases.get(c, [])) if cases.get(c) else float('nan') for _, _, cases in runs]
        checks = ' / '.join(('ok' if all(p for _, p in cases.get(c, [])) else 'FAIL') if cases.get(c) else '-' for _, _, cases in runs)
        red = [100 * (1 - v / vals[0]) for v in vals[1:]]
        summary[c] = {n: v for (n, _, _), v in zip(runs, vals)}
        lines.append(f'| {c} | ' + ' | '.join(f'{v:.0f}' for v in vals) + ' | ' + ' | '.join(f'{r:+.1f}%' for r in red) + f' | {checks} |')
    means = [mean(summary[c][n] for c in case_ids) for n, _, _ in runs]
    lines.append('| equal-weight mean | ' + ' | '.join(f'{v:.0f}' for v in means) + ' | ' + ' | '.join(f'{100*(1-v/means[0]):+.1f}%' for v in means[1:]) + ' | |')
    args.output.write_text(json.dumps({'runs': [(n, s) for n, s, _ in runs], 'cases': summary}, indent=1))
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
