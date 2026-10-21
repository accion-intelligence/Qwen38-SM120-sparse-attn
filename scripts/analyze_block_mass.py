"""Summarize block attention mass captures: oracle sparsity, estimator quality, structure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

COVERAGES = (0.90, 0.95, 0.99, 0.995)
KEPT_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
TAUS = (0.90, 0.95, 0.99)


def causal_counts(nB, device):
    return torch.arange(1, nB + 1, device=device, dtype=torch.float32)  # blocks available per query block


def oracle_kept(mass, coverages):
    """Per (head, q-block): blocks needed (sorted by true mass) to reach coverage; returns kept fractions."""
    H, nB, _ = mass.shape
    avail = causal_counts(nB, mass.device)
    sorted_mass, _ = torch.sort(mass, dim=-1, descending=True)
    cum = sorted_mass.cumsum(-1)
    out = {}
    for c in coverages:
        need = (cum < c).sum(-1).float() + 1  # [H, nB]
        need = torch.minimum(need, avail[None, :])
        out[c] = {'kept_fraction_global': (need.sum() / (avail.sum() * H)).item(),
                  'kept_fraction_mean_rows': (need / avail[None, :]).mean().item(),
                  'per_head_kept': (need.sum(1) / avail.sum()).tolist()}
    return out


def selection_coverage(mass, score, fractions, force_diag_sink=True):
    """True mass captured when keeping the top-f fraction of blocks ranked by score (per row)."""
    H, nB, _ = mass.shape
    dev = mass.device
    avail = causal_counts(nB, dev)
    score = score.float().clone()
    rows = torch.arange(nB, device=dev)
    if force_diag_sink:
        score[:, rows, rows] = float('inf')
        score[:, :, 0] = float('inf')
    order = torch.argsort(score, dim=-1, descending=True)
    ranked_mass = torch.gather(mass, -1, order)
    cum = ranked_mass.cumsum(-1)
    res = {}
    for f in fractions:
        keep = torch.clamp((avail * f).ceil(), min=2).long()  # at least diagonal + sink
        keep = torch.minimum(keep, avail.long())
        cov = torch.gather(cum, -1, (keep - 1)[None, :, None].expand(H, nB, 1)).squeeze(-1)
        res[f] = {'true_coverage_mean': cov.mean().item(), 'true_coverage_p10': cov.flatten().kthvalue(max(1, int(0.1 * cov.numel()))).values.item()}
    return res


def tau_rule(mass, est_prob, taus):
    """Adaptive rule on a probability estimator: keep top blocks until est cumsum >= tau."""
    H, nB, _ = mass.shape
    dev = mass.device
    avail = causal_counts(nB, dev)
    est = est_prob.float().clone()
    rows = torch.arange(nB, device=dev)
    est[:, rows, rows] += 1e9
    est[:, :, 0] += 1e9
    order = torch.argsort(est, dim=-1, descending=True)
    est_sorted = torch.gather(est_prob.float(), -1, order)
    est_sorted = est_sorted / est_sorted.sum(-1, keepdim=True).clamp_min(1e-12)
    true_sorted = torch.gather(mass, -1, order)
    est_cum = est_sorted.cumsum(-1)
    true_cum = true_sorted.cumsum(-1)
    res = {}
    for tau in taus:
        need = (est_cum < tau).sum(-1) + 1
        need = torch.minimum(torch.maximum(need, torch.full_like(need, 2)), avail.long()[None, :])
        cov = torch.gather(true_cum, -1, (need - 1)[..., None]).squeeze(-1)
        res[tau] = {'kept_fraction_global': (need.float().sum() / (avail.sum() * H)).item(),
                    'true_coverage_mean': cov.mean().item(),
                    'true_coverage_p10': cov.flatten().kthvalue(max(1, int(0.1 * cov.numel()))).values.item()}
    return res


def structure(mass):
    H, nB, _ = mass.shape
    dev = mass.device
    i = torch.arange(nB, device=dev)[:, None]
    j = torch.arange(nB, device=dev)[None, :]
    dist = (i - j)
    res = {}
    for w in (0, 1, 4, 16):
        res[f'band_le_{w}'] = mass[:, (dist <= w) & (dist >= 0).expand(nB, nB)].sum().item() / (H * nB) if False else (mass * ((dist >= 0) & (dist <= w)).float()[None]).sum().item() / (H * nB)
    res['sink_block0'] = mass[:, :, 0].sum().item() / (H * nB)
    col = mass.sum(1)  # [H, nB] total mass per key block
    col[:, 0] = 0
    top8 = col.topk(8, dim=-1).values.sum(-1) / nB  # fraction of (mass per row) in 8 strongest columns
    res['vertical_top8'] = top8.mean().item()
    return res


def needle(mass, meta, block):
    if 'needle_token_start' not in meta:
        return None
    nb = meta['needle_token_start'] // block
    qb = meta['question_token_start'] // block
    m = mass[:, qb:, nb]  # question rows -> needle block
    return {'needle_block': nb, 'question_rows_blocks': int(mass.shape[1] - qb),
            'mass_on_needle_block_per_head': m.mean(1).tolist(),
            'rank_of_needle_block_per_head': [(mass[h, qb:, :] > mass[h, qb:, nb][:, None]).float().sum(-1).mean().item() for h in range(mass.shape[0])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary = {}
    for pdir in sorted(args.input.iterdir()):
        if not (pdir / 'done.json').exists():
            continue
        meta = json.loads((pdir / 'done.json').read_text())
        block = meta['block']
        per_layer = {}
        agg = {'oracle': {c: [] for c in COVERAGES}, 'est': {}, 'structure': {}, 'needle': {}}
        for f in sorted(pdir.glob('*.pt')):
            d = torch.load(f, map_location=dev)
            mass = d['mass'].to(dev).float()
            o = oracle_kept(mass, COVERAGES)
            ests = {'rows_stride8': d['est_rows'].to(dev).float(), 'mean_pool': d['est_mean'].to(dev),
                    'antidiag_stride8': d['est_anti'].to(dev)}
            sel = {name: selection_coverage(mass, s, KEPT_FRACTIONS) for name, s in ests.items()}
            pairs = (block * block) // meta['stride']  # logits summed per block by the antidiagonal score
            tau = {'rows_stride8': tau_rule(mass, ests['rows_stride8'], TAUS),
                   'antidiag_softmax': tau_rule(mass, torch.softmax(ests['antidiag_stride8'].float(), -1).nan_to_num(0.0), TAUS),
                   'antidiag_meanlogit_softmax': tau_rule(mass, torch.softmax(ests['antidiag_stride8'].float() / pairs, -1).nan_to_num(0.0), TAUS),
                   'meanpool_softmax': tau_rule(mass, torch.softmax(ests['mean_pool'].float(), -1).nan_to_num(0.0), TAUS)}
            st = structure(mass)
            nd = needle(mass, meta, block)
            per_layer[d['layer']] = {'oracle': {str(c): {k: v for k, v in o[c].items()} for c in COVERAGES},
                                     'selection': {n: {str(k): v for k, v in r.items()} for n, r in sel.items()},
                                     'tau': {n: {str(k): v for k, v in r.items()} for n, r in tau.items()},
                                     'structure': st, 'needle': nd}
            for c in COVERAGES:
                agg['oracle'][c].append(o[c]['kept_fraction_global'])
            for n, r in sel.items():
                agg['est'].setdefault(n, {f: [] for f in KEPT_FRACTIONS})
                for fr in KEPT_FRACTIONS:
                    agg['est'][n][fr].append(r[fr]['true_coverage_mean'])
            for n, r in tau.items():
                agg['est'].setdefault('tau:' + n, {t: [] for t in TAUS})
                for t in TAUS:
                    agg['est']['tau:' + n][t].append((r[t]['kept_fraction_global'], r[t]['true_coverage_mean']))
            for k, v in st.items():
                agg['structure'].setdefault(k, []).append(v)
            del mass, ests, d
        mean = lambda xs: sum(xs) / len(xs)
        summary[pdir.name] = {
            'meta': meta,
            'oracle_kept_fraction_mean_over_layers': {str(c): mean(v) for c, v in agg['oracle'].items()},
            'oracle_kept_fraction_per_layer_99': {name: v['oracle']['0.99']['kept_fraction_global'] for name, v in per_layer.items()},
            'estimator_true_coverage_at_kept_fraction': {n: {str(f): mean(v) for f, v in r.items()} for n, r in agg['est'].items() if not n.startswith('tau:')},
            'tau_rule_kept_and_coverage': {n: {str(t): [mean([x[0] for x in v]), mean([x[1] for x in v])] for t, v in r.items()} for n, r in agg['est'].items() if n.startswith('tau:')},
            'structure_mean_over_layers': {k: mean(v) for k, v in agg['structure'].items()},
            'per_layer': per_layer,
        }
        print(pdir.name, 'oracle kept@99%:', f"{summary[pdir.name]['oracle_kept_fraction_mean_over_layers']['0.99']:.3f}", flush=True)
    args.output.write_text(json.dumps(summary, indent=1))
    # markdown
    lines = ['| prompt | oracle kept @90% | @95% | @99% | @99.5% | rows-sample τ=.99 kept/cov | antidiag τ=.99 kept/cov | band≤1 | band≤16 | sink | top8 cols |', '|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for name, s in summary.items():
        o = s['oracle_kept_fraction_mean_over_layers']; st = s['structure_mean_over_layers']; t = s['tau_rule_kept_and_coverage']
        r = t['tau:rows_stride8']['0.99']; a = t['tau:antidiag_softmax']['0.99']
        lines.append(f"| {name} | {o['0.9']:.3f} | {o['0.95']:.3f} | {o['0.99']:.3f} | {o['0.995']:.3f} | {r[0]:.3f}/{r[1]:.3f} | {a[0]:.3f}/{a[1]:.3f} | {st['band_le_1']:.3f} | {st['band_le_16']:.3f} | {st['sink_block0']:.3f} | {st['vertical_top8']:.3f} |")
    lines.append('')
    lines.append('Per-layer oracle kept fraction at 99% coverage, and needle retrieval (docs_needle: mean rank of the needle block among causal blocks for question rows, per layer min/median/max over heads):')
    lines.append('')
    lines.append('| prompt | kept@99% per layer (layer index: value) | needle rank min/median/max | mass on needle block max head |')
    lines.append('|---|---|---|---|')
    for name, s in summary.items():
        pl = s['oracle_kept_fraction_per_layer_99']
        per = ', '.join(f"{k.split('layers.')[1].split('.')[0]}:{v:.2f}" for k, v in pl.items())
        nd = [v['needle'] for v in s['per_layer'].values() if v.get('needle')]
        if nd:
            ranks = [r for v in nd for r in v['rank_of_needle_block_per_head']]
            ranks.sort()
            masses = [m for v in nd for m in v['mass_on_needle_block_per_head']]
            ndtxt = f"{ranks[0]:.1f}/{ranks[len(ranks)//2]:.1f}/{ranks[-1]:.1f}"
            mtxt = f"{max(masses):.3f}"
        else:
            ndtxt = mtxt = '-'
        lines.append(f'| {name} | {per} | {ndtxt} | {mtxt} |')
    lines.append('')
    lines.append('Adaptive τ rules (kept fraction / true coverage), mean over layers:')
    lines.append('')
    lines.append('| prompt | estimator | τ=0.90 | τ=0.95 | τ=0.99 |')
    lines.append('|---|---|---|---|---|')
    for name, s in summary.items():
        for est, r in s['tau_rule_kept_and_coverage'].items():
            lines.append(f"| {name} | {est[4:]} | " + ' | '.join(f"{r[t][0]:.3f}/{r[t][1]:.3f}" for t in ('0.9', '0.95', '0.99')) + ' |')
    lines.append('')
    lines.append('Ranking quality: true coverage when keeping a fixed fraction of causal blocks chosen by each estimator (diagonal and block 0 always kept), mean over layers:')
    lines.append('')
    lines.append('| prompt | estimator | 5% | 10% | 20% | 30% | 50% |')
    lines.append('|---|---|---|---|---|---|---|')
    for name, s in summary.items():
        for est, r in s['estimator_true_coverage_at_kept_fraction'].items():
            lines.append(f"| {name} | {est} | " + ' | '.join(f"{r[f]:.3f}" for f in ('0.05', '0.1', '0.2', '0.3', '0.5')) + ' |')
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
