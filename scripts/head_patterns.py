"""Per-(layer, head) attention structure across prompts: local band share, kept@95%, consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def head_stats(mass, band_blocks=16):
    H, nB, _ = mass.shape
    dev = mass.device
    i = torch.arange(nB, device=dev)[:, None]
    j = torch.arange(nB, device=dev)[None, :]
    dist = i - j
    band = ((dist >= 0) & (dist <= band_blocks)).float()[None]
    local = (mass * band).sum((1, 2)) / nB                       # [H] share of mass within band
    sink = mass[:, :, 0].sum(1) / nB
    avail = torch.arange(1, nB + 1, device=dev, dtype=torch.float32)
    srt, _ = torch.sort(mass, dim=-1, descending=True)
    cum = srt.cumsum(-1)
    kept95 = torch.minimum((cum < 0.95).sum(-1).float() + 1, avail[None]).sum(1) / avail.sum()
    kept99 = torch.minimum((cum < 0.99).sum(-1).float() + 1, avail[None]).sum(1) / avail.sum()
    return local, sink, kept95, kept99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--band-blocks', type=int, default=16)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    table = {}  # prompt -> layer -> dict(head stats lists)
    for pdir in sorted(args.input.iterdir()):
        if not (pdir / 'done.json').exists():
            continue
        table[pdir.name] = {}
        for f in sorted(pdir.glob('*.pt')):
            d = torch.load(f, map_location=dev)
            local, sink, k95, k99 = head_stats(d['mass'].to(dev).float(), args.band_blocks)
            table[pdir.name][d['layer']] = {'local': local.tolist(), 'sink': sink.tolist(),
                                            'kept95': k95.tolist(), 'kept99': k99.tolist()}
    prompts = list(table)
    layers = sorted(next(iter(table.values())))
    lines = []
    # consistency of per-head local share across prompts
    import itertools
    rows = []
    for a, b in itertools.combinations(prompts, 2):
        xa = torch.tensor([table[a][l]['local'] for l in layers]).flatten()
        xb = torch.tensor([table[b][l]['local'] for l in layers]).flatten()
        corr = torch.corrcoef(torch.stack([xa, xb]))[0, 1].item()
        rows.append(f'| {a} vs {b} | {corr:.3f} |')
    lines += ['Correlation of per-head local-band share between prompts (384 heads):', '', '| pair | corr |', '|---|---:|'] + rows + ['']
    # head classification counts per prompt
    lines += [f'Heads by local-band share (band = {args.band_blocks} blocks = {args.band_blocks*64} tokens), out of 384 heads; and heads needing <=25% / >75% blocks for 95% coverage:', '',
              '| prompt | local>0.9 | local>0.8 | local<0.5 | kept95<=0.25 | kept95>0.75 | mean kept95 | mean kept99 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in prompts:
        loc = torch.tensor([table[p][l]['local'] for l in layers]).flatten()
        k95 = torch.tensor([table[p][l]['kept95'] for l in layers]).flatten()
        k99 = torch.tensor([table[p][l]['kept99'] for l in layers]).flatten()
        lines.append(f'| {p} | {(loc>0.9).sum().item()} | {(loc>0.8).sum().item()} | {(loc<0.5).sum().item()} | {(k95<=0.25).sum().item()} | {(k95>0.75).sum().item()} | {k95.mean():.3f} | {k99.mean():.3f} |')
    lines.append('')
    # per-layer head summary for the longest docs prompt
    pick = [p for p in prompts if p.startswith('docs_needle')]
    pick = sorted(pick, key=lambda s: int(s.split('-')[1]))[-1] if pick else prompts[-1]
    lines += [f'Per-layer head detail for {pick}: local share min/median/max, kept95 min/median/max:', '', '| layer | local min/med/max | kept95 min/med/max |', '|---|---|---|']
    for l in layers:
        loc = torch.tensor(table[pick][l]['local']); k = torch.tensor(table[pick][l]['kept95'])
        lines.append(f"| {l.split('layers.')[1].split('.')[0]} | {loc.min():.2f}/{loc.median():.2f}/{loc.max():.2f} | {k.min():.2f}/{k.median():.2f}/{k.max():.2f} |")
    args.output.write_text(json.dumps(table))
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
