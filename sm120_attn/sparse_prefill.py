"""Dynamic block-sparse causal prefill attention for long contexts (SM120 prototype).

Per full-attention layer and prefill chunk: mean-pooled q/k block scores,
per-head top-mass (tau) block selection, union over the query heads of each
KV head (GQA), then FlashInfer block-sparse attention (FA2 template, 64x64
blocks, causal alignment handled by placing the diagonal block last).
"""
from __future__ import annotations

import json
import math
import os

import torch

BLOCK = 64


class SparsePrefill:
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, device,
                 tau: float = 0.95, min_kv: int = 8192, min_q: int = 1024,
                 workspace_bytes: int = 256 << 20, kernel: str = 'sm120fp8', fp8_pipeline: str = 'tma'):
        self.kernel = kernel
        self.fp8_pipeline = fp8_pipeline     # 'tma' (warp-specialised TMA/mbarrier fp8 kernel) | 'cpasync'
        self.H, self.Hk, self.D = num_heads, num_kv_heads, head_dim
        self.g = num_heads // num_kv_heads
        self.tau, self.min_kv, self.min_q = tau, min_kv, min_q
        self.scale = head_dim ** -0.5
        self.wrapper = None
        if kernel == 'flashinfer':      # retired reference path; needs flashinfer-python
            import flashinfer
            self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            self.wrapper = flashinfer.BlockSparseAttentionWrapper(self.workspace, backend='fa2')
        self.stats = {'calls': 0, 'blocks_total': 0, 'blocks_kept': 0, 'plan_ms': 0.0, 'run_ms': 0.0, 'select_ms': 0.0}
        self._kept_acc = None
        self.timing = False

    def eligible(self, q_len: int, kv_len: int) -> bool:
        return q_len >= self.min_q and kv_len >= self.min_kv

    @torch.no_grad()
    def block_masks(self, q: torch.Tensor, k, q_start: int, valid=None, union: bool = True, km=None):
        """q [n, H, D]; k [Hk, N, D] (head-major) or [N, Hk, D] (NHD) with N a multiple of BLOCK;
        q_start = absolute position of q[0]. Returns bool masks [Hk, MB, NB] (union over the query
        heads of each KV head) or [H, MB, NB] per query head when union=False.
        """
        n = q.shape[0]
        if km is not None:                                   # cached block means [Hk, NB, D] (fp32)
            NB = km.shape[1]
            head_major = None
        else:
            head_major = k.shape[0] == self.Hk and k.shape[1] != self.Hk
            N = k.shape[1] if head_major else k.shape[0]
            NB = N // BLOCK
        MB = -(-n // BLOCK)
        dev = q.device
        assert n % BLOCK == 0 and q_start % BLOCK == 0, 'callers pad q to whole blocks'
        if valid is None:
            valid = (0, n)
        rows = torch.arange(n, device=dev)
        real = ((rows >= valid[0]) & (rows < valid[0] + valid[1])).float()
        counts = real.view(MB, BLOCK).sum(1).clamp_min(1.0)
        qm = (q.float() * real[:, None, None]).view(MB, BLOCK, self.H, self.D).sum(1) / counts[:, None, None]
        if km is None and head_major:
            km = k.view(self.Hk, NB, BLOCK, self.D).float().mean(2)                  # [Hk, NB, D]
        elif km is None:
            km = k.view(NB, BLOCK, self.Hk, self.D).float().mean(1).transpose(0, 1)  # [Hk, NB, D]
        gi = (q_start // BLOCK) + torch.arange(MB, device=dev)                       # diagonal block per q block
        j = torch.arange(NB, device=dev)
        future = j[None, :] > gi[:, None]                                            # [MB, NB]
        qg = qm.view(MB, self.Hk, self.g, self.D)
        logits = torch.einsum('mhgd,hnd->hmgn', qg, km) * self.scale                 # [Hk, MB, g, NB]
        logits.masked_fill_(future[None, :, None, :], float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        srt, order = torch.sort(probs, dim=-1, descending=True)
        keep = (srt.cumsum(-1) < self.tau).sum(-1, keepdim=True) + 1                 # blocks per (h, m, head)
        sel = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, order, j[None, None, None, :] < keep)
        if union:
            masks = sel.any(2)                                                       # [Hk, MB, NB]
        else:
            masks = sel.permute(0, 2, 1, 3).reshape(self.H, MB, NB)                  # [H, MB, NB], head = kvh * g + i
        masks[:, torch.arange(MB, device=dev), gi] = True                            # diagonal
        masks[:, :, 0] = True                                                        # sink
        masks &= ~future[None]
        return masks

    @torch.no_grad()
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_start: int,
                out: torch.Tensor | None = None) -> torch.Tensor:
        """q [n, H, D] bf16 at absolute positions q_start..; k, v [Hk, kv_len, D] bf16 (contiguous per head)
        for the FlashInfer kernel, or [kv_len, Hk, D] (NHD) for the SM120 kernel."""
        if self.kernel == 'sm120':
            return self.forward_sm120(q, k, v, q_start, out)
        if self.kernel == 'sm120fp8':
            return self.forward_sm120fp8(q, k, v, q_start, out)
        n = q.shape[0]
        kv_len = q_start + n
        assert k.shape[1] >= kv_len, (q_start, n, k.shape)
        dev = q.device
        front = q_start % BLOCK
        end = (-(front + n)) % BLOCK
        kpad = (-k.shape[1]) % BLOCK
        if kpad:
            k = torch.nn.functional.pad(k, (0, 0, 0, kpad))
            v = torch.nn.functional.pad(v, (0, 0, 0, kpad))
        N = k.shape[1]
        NB = N // BLOCK
        if front or end:
            q = torch.cat([q.new_zeros(front, self.H, self.D), q, q.new_zeros(end, self.H, self.D)])
        M = q.shape[0]
        MB = M // BLOCK
        qs = q_start - front
        assert qs + M <= N
        t0 = self._tick()
        masks = self.block_masks(q, k, qs, valid=(front, n))               # [Hk, MB, NB]
        # One block-sparse problem for all KV heads: row blocks and key blocks are group-major.
        counts = masks.sum(2)                                              # [Hk, MB]
        indptr = torch.zeros(self.Hk * MB + 1, dtype=torch.int32, device=dev)
        indptr[1:] = counts.flatten().cumsum(0)
        nz = masks.nonzero()                                               # (g, m, j) row-major: ascending j, diagonal last
        indices = (nz[:, 0] * NB + nz[:, 2]).to(torch.int32)
        t1 = self._tick()
        self.wrapper.plan(indptr, indices, self.Hk * M, self.Hk * N, BLOCK, BLOCK, self.g, 1, self.D,
                          causal=True, sm_scale=self.scale, q_data_type=q.dtype, kv_data_type=k.dtype,
                          o_data_type=q.dtype)
        t2 = self._tick()
        q_all = q.view(M, self.Hk, self.g, self.D).permute(1, 0, 2, 3).reshape(self.Hk * M, self.g, self.D)
        o_all = self.wrapper.run(q_all, k.reshape(self.Hk * N, 1, self.D), v.reshape(self.Hk * N, 1, self.D))
        o = o_all.view(self.Hk, M, self.g, self.D).permute(1, 0, 2, 3).reshape(M, self.H, self.D)[front:front + n]
        t3 = self._tick()
        if self.timing:
            self.stats['select_ms'] += t1 - t0
            self.stats['plan_ms'] += t2 - t1
            self.stats['run_ms'] += t3 - t2
        self.stats['calls'] += 1
        self.stats['blocks_total'] += int(self.Hk * ((qs // BLOCK) * MB + MB * (MB + 1) // 2))
        self.stats['blocks_kept'] += int(indices.numel())
        if out is None:
            return o
        out.copy_(o)
        return out

    @torch.no_grad()
    def forward_sm120(self, q, k, v, q_start, out=None, km=None):
        """Per-query-head block-sparse attention with the custom SM120 kernel.

        k, v: NHD [T, Hk, D] with T >= q_start + n (rows beyond the sequence are ignored);
        km: optional cached block means [Hk, ceil((q_start+n)/BLOCK), D] (fp32). No host syncs.
        """
        from .kernels.kai_fa_bf16 import attention_sm120
        n = q.shape[0]
        kv_len = q_start + n
        assert k.shape[0] >= kv_len
        front = q_start % BLOCK
        end = (-(front + n)) % BLOCK
        kpad = (-k.shape[0]) % BLOCK
        if kpad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, kpad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, kpad))
        if front or end:
            q = torch.cat([q.new_zeros(front, self.H, self.D), q, q.new_zeros(end, self.H, self.D)])
        M = q.shape[0]
        MB = M // BLOCK
        qs = q_start - front
        t0 = self._tick()
        if km is None:
            nb = -(-kv_len // BLOCK)
            km = k[:nb * BLOCK].view(nb, BLOCK, self.Hk, self.D).float().mean(1).transpose(0, 1)
        if FP8_EMULATE:
            # Numerics experiment only: round attention operands through e4m3 (BF16 kernel unchanged).
            q, k, v = _fp8_round(q, FP8_EMULATE, 'q'), _fp8_round(k, FP8_EMULATE, 'k'), _fp8_round(v, FP8_EMULATE, 'v')
        masks = self.block_masks(q, None, qs, valid=(front, n), union=False, km=km)   # [H, MB, NB]
        counts = masks.sum(2).to(torch.int32).flatten()
        lists = torch.argsort((~masks).to(torch.int8), dim=-1, stable=True).to(torch.int32).reshape(self.H * MB, -1)
        t1 = self._tick()
        o = attention_sm120(q, k, v, qs, counts=counts, lists=lists)[front:front + n]
        t2 = self._tick()
        if self.timing:
            self.stats['select_ms'] += t1 - t0
            self.stats['run_ms'] += t2 - t1
        self.stats['calls'] += 1
        self.stats['blocks_total'] += int(self.H * ((qs // BLOCK) * MB + MB * (MB + 1) // 2))
        self._kept_acc = counts.sum() if self._kept_acc is None else self._kept_acc + counts.sum()
        if out is None:
            return o
        out.copy_(o)
        return out

    @torch.no_grad()
    def forward_sm120fp8(self, q, k, v, q_start, out=None, km=None, scales=None):
        """FP8 (e4m3) SM120 kernel, 128-row query blocks; k, v NHD [T, Hk, D] bf16, quantised here.

        scales: (sq, sk, sv) python floats (per-request, per-layer, cached by the caller); without
        them per-call amax scales are used (three device syncs).
        """
        from .kernels.kai_fa_fp8 import attention_sm120_fp8, permute_vt_keys
        from .kernels.kai_fa_fp8_tma import attention_sm120_fp8_tma
        run_kernel = attention_sm120_fp8_tma if self.fp8_pipeline == 'tma' else attention_sm120_fp8
        BM = 128
        n = q.shape[0]
        kv_len = q_start + n
        assert k.shape[0] >= kv_len
        front = q_start % BM
        end = (-(front + n)) % BM
        if front or end:
            q = torch.cat([q.new_zeros(front, self.H, self.D), q, q.new_zeros(end, self.H, self.D)])
        M = q.shape[0]
        MB = M // BLOCK
        qs = q_start - front
        # keys must cover the padded query range (padded rows are discarded; extra keys are future, masked)
        need = max(-(-k.shape[0] // BLOCK) * BLOCK, qs + M)
        if need > k.shape[0]:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, need - k.shape[0]))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, need - v.shape[0]))
        t0 = self._tick()
        if km is None:
            nb = -(-kv_len // BLOCK)
            km = k[:nb * BLOCK].view(nb, BLOCK, self.Hk, self.D).float().mean(1).transpose(0, 1)
        nb_need = need // BLOCK
        if km.shape[1] < nb_need:                     # padded query range reaches beyond the real keys
            km = torch.nn.functional.pad(km, (0, 0, 0, nb_need - km.shape[1]))
        masks = self.block_masks(q, None, qs, valid=(front, n), union=False, km=km)   # [H, MB(64), NB]
        masks = masks.view(self.H, MB // 2, 2, -1).any(2)                            # 128-row blocks
        counts = masks.sum(2).to(torch.int32).flatten()
        lists = torch.argsort((~masks).to(torch.int8), dim=-1, stable=True).to(torch.int32).reshape(self.H * (MB // 2), -1)
        if scales is None:
            scales = tuple(float(t.float().abs().amax().clamp_min(1e-6).item()) / 448.0 for t in (q, k, v))
        sq, sk, sv = scales
        # bf16 arithmetic (no fp32 temporaries): scales are per request so clipping at 448 is by construction
        q8 = (q * (1.0 / sq)).to(torch.float8_e4m3fn)
        k8 = (k * (1.0 / sk)).to(torch.float8_e4m3fn)
        # keys permuted inside each 32-key block (kernel builds P's A fragment in registers; see VT_KEY_PERM)
        vT8 = permute_vt_keys((v * (1.0 / sv)).to(torch.float8_e4m3fn).permute(1, 2, 0))
        t1 = self._tick()
        o = run_kernel(q, k, v, qs, counts=counts, lists=lists, prequant=(q8, sq, k8, sk, vT8, sv),
                       m_block=BM)[front:front + n]
        t2 = self._tick()
        if self.timing:
            self.stats['select_ms'] += t1 - t0
            self.stats['run_ms'] += t2 - t1
        self.stats['calls'] += 1
        self.stats['blocks_total'] += int(self.H * ((qs // BLOCK) * MB + MB * (MB + 1) // 2))
        self._kept_acc = counts.sum() * 2 if self._kept_acc is None else self._kept_acc + counts.sum() * 2
        if out is None:
            return o
        out.copy_(o)
        return out

    def kept_fraction(self):
        """Blocks kept / causal blocks over all SM120 calls so far (one device sync)."""
        if self._kept_acc is None or not self.stats['blocks_total']:
            return None
        return int(self._kept_acc.item()) / self.stats['blocks_total']

    def _tick(self):
        if not self.timing:
            return 0.0
        torch.cuda.synchronize()
        import time
        return time.time() * 1000.0

def reference(q, k, v, q_start, masks=None):
    """fp32 reference: causal attention of q [n,H,D] (positions q_start..) over k,v [Hk,N,D], optional block masks."""
    n, H, D = q.shape
    Hk, N, _ = k.shape
    g = H // Hk
    pos_q = q_start + torch.arange(n, device=q.device)
    pos_k = torch.arange(N, device=q.device)
    allowed = pos_k[None, :] <= pos_q[:, None]                                        # [n, N]
    outs = []
    for h in range(H):
        s = (q[:, h].float() @ k[h // g].float().T) * D ** -0.5
        mask = allowed.clone()
        if masks is not None:
            bm = masks[h // g]                                                       # [MB, NB]
            mask &= bm.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :N]
        s.masked_fill_(~mask, float('-inf'))
        outs.append(torch.softmax(s, -1) @ v[h // g].float())
    return torch.stack(outs, 1)


if __name__ == '__main__':
    torch.manual_seed(0)
    dev = 'cuda'
    H, Hk, D = 24, 4, 256

    def make(n, kv_len, peaky=3.0):
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(Hk, kv_len, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(Hk, kv_len, D, device=dev, dtype=torch.bfloat16)
        # block-structured: 10% of key blocks are "hot": keys share a per-head direction that queries point to
        hot = (torch.rand(kv_len // BLOCK + 1, device=dev) < 0.1).repeat_interleave(BLOCK)[:kv_len]
        u = torch.randn(Hk, 1, D, device=dev)
        k = (k.float() * 0.5 + hot[None, :, None] * u * peaky).to(torch.bfloat16)
        q = (q.float() + u.repeat_interleave(H // Hk, 0).transpose(0, 1) * 1.0).to(torch.bfloat16)
        return q, k, v

    for n, kv_len, tau in ((512, 2048, 1.0), (512, 2048, 0.9), (1000, 4024, 0.9), (777, 3001, 0.95), (64, 4096, 0.9)):
        sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0)
        q, k, v = make(n, kv_len)
        q_start = kv_len - n
        o = sp.forward(q, k, v, q_start)
        ref_dense = reference(q, k, v, q_start)
        front = q_start % BLOCK; end = (-(front + n)) % BLOCK; kpad = (-kv_len) % BLOCK
        kp = torch.nn.functional.pad(k, (0, 0, 0, kpad)); vp = torch.nn.functional.pad(v, (0, 0, 0, kpad))
        qp = torch.cat([q.new_zeros(front, H, D), q, q.new_zeros(end, H, D)])
        masks = sp.block_masks(qp, kp, q_start - front, valid=(front, n))
        ref_masked = reference(qp, kp, vp, q_start - front, masks)[front:front + n]
        e_mask = (o.float() - ref_masked).abs().max().item() / ref_masked.abs().max().item()
        e_dense = (o.float() - ref_dense).norm().item() / ref_dense.norm().item()
        print(f'n={n:5d} kv={kv_len:5d} q_start={q_start:5d} tau={tau}: kept {sp.stats["blocks_kept"]/sp.stats["blocks_total"]:.3f} | max err vs masked ref {e_mask:.2e} | rel L2 vs dense {e_dense:.3e}')
    for tau in (0.95, 0.99, 1.0):
        n, kv_len = 8192, 65536
        q, k, v = make(n, kv_len)
        sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0)
        for _ in range(2): sp.forward(q, k, v, kv_len - n)
        sp.timing = True; sp.stats.update(plan_ms=0.0, run_ms=0.0, select_ms=0.0, blocks_total=0, blocks_kept=0)
        iters = 5
        for _ in range(iters): sp.forward(q, k, v, kv_len - n)
        st = sp.stats
        print(f'n={n} kv={kv_len} tau={tau}: kept {st["blocks_kept"]/st["blocks_total"]:.3f} | select {st["select_ms"]/iters:.2f} ms, plan {st["plan_ms"]/iters:.2f} ms, run {st["run_ms"]/iters:.2f} ms | dense FA2 paged reference 40.5 ms')


# ---------------------------------------------------------------------------
# vLLM glue: dynamic block-sparse path for long prefill chunks in the
# FlashAttention backend. Enabled by SM120_SPARSE_ATTN (default on in this
# candidate); SM120_SPARSE_TAU, SM120_SPARSE_MIN_KV, SM120_SPARSE_MIN_Q tune it.
# ---------------------------------------------------------------------------
_CONFIG = None
_STATE: dict = {}
FP8_EMULATE = os.environ.get('SM120_SPARSE_FP8_EMULATE', '')      # '' | 'tensor' | 'token' (experiment switch)


def _fp8_round(x, mode, which):
    """Round a [rows, heads, D] bf16 tensor through float8_e4m3fn with amax scaling.

    mode 'tensor': one scale per tensor (vLLM's fp8 KV cache with calculated scales);
    mode 'token': per (row, head) scales for q/k and per (head, channel) scales for v.
    """
    xf = x.float()
    if mode == 'tensor':
        amax = xf.abs().amax().clamp_min(1e-6)
    elif which == 'v':
        amax = xf.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)            # per head, channel
    else:
        amax = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-6)            # per row, head
    scale = amax / 448.0
    return ((xf / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
_DEBUG_LEFT = [int(os.environ.get('SM120_SPARSE_DEBUG', '0'))]


def _debug(msg):
    if _DEBUG_LEFT[0] > 0:
        _DEBUG_LEFT[0] -= 1
        print('[sm120-sparse] ' + msg, flush=True)


def config():
    """SM120_SPARSE_* environment variables, with optional defaults from the JSON file named by
    SM120_SPARSE_CONFIG (useful when a launcher does not forward environment variables)."""
    global _CONFIG
    if _CONFIG is None:
        file_cfg = {}
        path = os.environ.get('SM120_SPARSE_CONFIG')
        if path:
            with open(path) as f:
                file_cfg = json.load(f)
        def pick(env, key, default, cast):
            if env in os.environ:
                return cast(os.environ[env])
            return cast(file_cfg.get(key, default))
        _CONFIG = {
            'enabled': pick('SM120_SPARSE_ATTN', 'enabled', 1, lambda v: str(v) not in ('0', 'false', 'False')),
            'tau': pick('SM120_SPARSE_TAU', 'tau', 0.95, float),
            'min_kv': pick('SM120_SPARSE_MIN_KV', 'min_kv', 8192, int),
            'min_q': pick('SM120_SPARSE_MIN_Q', 'min_q', 1024, int),
            'kernel': pick('SM120_SPARSE_KERNEL', 'kernel', 'sm120fp8', str),
            'fp8_pipeline': pick('SM120_SPARSE_FP8_PIPELINE', 'fp8_pipeline', 'tma', str),
        }
        print('[sm120-sparse] config %r' % (_CONFIG,), flush=True)
    return _CONFIG


def _state(num_heads, num_kv_heads, head_dim, device):
    key = (num_heads, num_kv_heads, head_dim, str(device))
    if key not in _STATE:
        cfg = config()
        _STATE[key] = SparsePrefill(num_heads, num_kv_heads, head_dim, device, tau=cfg['tau'],
                                    min_kv=cfg['min_kv'], min_q=cfg['min_q'], kernel=cfg['kernel'],
                                    fp8_pipeline=cfg['fp8_pipeline'])
    return _STATE[key]


_KMEAN_CACHE: dict = {}
_KMEAN_MAX_ENTRIES = 128


def _cached_block_means(layer_key, first_page, key_chunk, q_start, kv_len, key_all=None):
    """Maintain per-(layer, request) K block means [Hk, NB, D] (fp32), updated from the chunk's keys.

    A request is identified by its first physical page. An entry is only trusted when it was built
    contiguously by this request (its last chunk ended exactly at q_start); otherwise - new request,
    or a request whose first chunks were too short for the sparse path and whose first page was
    recycled from an earlier request - the entry is rebuilt from the full key history `key_all`
    [kv_len, Hk, D] (always available: the glue gathers the whole KV history of the request).
    Chunk boundaries unaligned to BLOCK make the boundary block's mean use only the rows seen so far.
    """
    n, Hk, D = key_chunk.shape
    cache_key = (layer_key, first_page)
    entry = _KMEAN_CACHE.get(cache_key)
    b0, b1 = q_start // BLOCK, -(-(q_start + n) // BLOCK)
    if entry is None or q_start == 0 or entry['nb'] < b0 or entry.get('last_kv') != q_start:
        if len(_KMEAN_CACHE) >= _KMEAN_MAX_ENTRIES:
            _KMEAN_CACHE.pop(next(iter(_KMEAN_CACHE)))
        cap = max(4096, b1)
        entry = {'km': torch.zeros(Hk, cap, D, dtype=torch.float32, device=key_chunk.device), 'nb': 0, 'last_kv': 0}
        _KMEAN_CACHE[cache_key] = entry
        if q_start > 0:
            if key_all is None:
                raise RuntimeError('stale block-mean cache entry and no key history to rebuild it')
            # rebuild the means of the blocks before this chunk from the request's own history
            hist = key_all[:q_start]
            full, rem = q_start // BLOCK, q_start % BLOCK
            if full:
                entry['km'][:, :full] = hist[:full * BLOCK].float().view(full, BLOCK, Hk, D).mean(1).transpose(0, 1)
            if rem:   # partial block: the chunk below completes it from `front` rows of its own data
                entry['km'][:, full] = hist[full * BLOCK:].float().sum(0) / rem
            entry['nb'] = full + (1 if rem else 0)
            key_chunk = key_chunk if rem == 0 else torch.cat([hist[full * BLOCK:], key_chunk], 0)
            if rem:
                q_start, n = full * BLOCK, n + rem
                b0 = full
    if entry['km'].shape[1] < b1:
        grown = torch.zeros(Hk, max(b1, 2 * entry['km'].shape[1]), D, dtype=torch.float32, device=key_chunk.device)
        grown[:, :entry['km'].shape[1]] = entry['km']
        entry['km'] = grown
    front = q_start - b0 * BLOCK
    rows = (b1 - b0) * BLOCK
    padded = key_chunk.new_zeros(rows, Hk, D, dtype=torch.float32)
    padded[front:front + n] = key_chunk.float()
    valid = torch.zeros(rows, device=key_chunk.device, dtype=torch.float32)
    valid[front:front + n] = 1.0
    cnt = valid.view(b1 - b0, BLOCK).sum(1).clamp_min(1.0)
    entry['km'][:, b0:b1] = (padded.view(b1 - b0, BLOCK, Hk, D).sum(1) / cnt[:, None, None]).transpose(0, 1)
    entry['nb'] = b1
    entry['last_kv'] = kv_len
    return entry['km'][:, :b1]


_SCALE_CACHE: dict = {}
SCALE_MARGIN = 1.5


def _cached_scales(layer_key, first_page, q, k, v, q_start):
    """Per-(layer, request) e4m3 scales (amax/448 x margin) from the request's first sparse chunk.

    k, v are the request's full history so far; the entry is trusted only if its last chunk ended at
    q_start (continuity), so a recycled first page of another request never supplies a stale scale.
    """
    key = (layer_key, first_page)
    entry = _SCALE_CACHE.get(key)
    if q_start == 0 or entry is None or entry['last_kv'] != q_start:
        if len(_SCALE_CACHE) >= _KMEAN_MAX_ENTRIES:
            _SCALE_CACHE.pop(next(iter(_SCALE_CACHE)))
        amax = torch.stack([t.float().abs().amax() for t in (q, k, v)]).clamp_min(1e-6).tolist()
        entry = {'scales': tuple(a * SCALE_MARGIN / 448.0 for a in amax), 'last_kv': 0}
        _SCALE_CACHE[key] = entry
    entry['last_kv'] = q_start + q.shape[0]
    return entry['scales']


@torch.no_grad()
def try_sparse_forward(impl, query, key, key_cache, value_cache, output, attn_metadata) -> bool:
    """Block-sparse attention for the long prefill chunks of this batch; other requests in the
    same batch (short chunks, decode steps) get a per-request dense FlashAttention call.

    query/key/output: this batch's [num_actual_tokens, ...] tensors; key_cache/value_cache:
    [num_blocks, page, Hk, D]. Returns False (nothing written) when no request qualifies.
    """
    cfg = config()
    if not cfg['enabled']:
        return False
    window = getattr(impl, 'sliding_window', None)
    windowed = window is not None and any(int(w) >= 0 for w in window)   # full attention is (-1, -1)
    if attn_metadata.causal is not True or windowed:
        _debug('fallback: causal=%r window=%r' % (attn_metadata.causal, window))
        return False
    # Host-side checks first: no device-to-host copies during CUDA graph capture (decode-sized
    # batches are captured at startup) and no sync for batches that cannot qualify.
    if attn_metadata.max_query_len < cfg['min_q'] or attn_metadata.max_seq_len < cfg['min_kv']:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    qsl = attn_metadata.query_start_loc.tolist()
    seq_lens = attn_metadata.seq_lens.tolist()
    nreq = len(seq_lens)
    eligible = [r for r in range(nreq) if qsl[r + 1] - qsl[r] >= cfg['min_q'] and seq_lens[r] >= cfg['min_kv']]
    if not eligible:
        return False
    _debug('sparse path: %d/%d requests, q_lens=%s kv_lens=%s' % (len(eligible), nreq, [qsl[i + 1] - qsl[i] for i in range(nreq)], seq_lens))
    H, D = query.shape[1], query.shape[2]
    Hk = key_cache.shape[2]
    sp = _state(H, Hk, D, query.device)
    page = key_cache.shape[1]
    kc = key_cache.permute(2, 0, 1, 3)       # [Hk, num_blocks, page, D] view
    vc = value_cache.permute(2, 0, 1, 3)
    bt = attn_metadata.block_table
    first_pages = bt[:nreq, 0].tolist() if sp.kernel in ('sm120', 'sm120fp8') else None
    for r in range(nreq):
        kv_len = seq_lens[r]
        n = qsl[r + 1] - qsl[r]
        q_r = query[qsl[r]:qsl[r + 1]]
        o_r = output[qsl[r]:qsl[r + 1]]
        if r not in eligible:
            # short chunk or decode step of another request in the same batch: dense FlashAttention per request
            _dense_request(impl, q_r, key_cache, value_cache, o_r, bt[r], n, kv_len)
            continue
        npages = -(-kv_len // page)
        pages = bt[r, :npages].long()
        if sp.kernel in ('sm120', 'sm120fp8'):
            k = key_cache[pages].reshape(npages * page, Hk, D)    # one gather copy, NHD
            v = value_cache[pages].reshape(npages * page, Hk, D)
            km = _cached_block_means(id(impl), first_pages[r], key[qsl[r]:qsl[r + 1]], kv_len - n, kv_len, key_all=k)
            if sp.kernel == 'sm120fp8':
                scales = _cached_scales(id(impl), first_pages[r], q_r, k[:kv_len], v[:kv_len], kv_len - n)
                sp.forward_sm120fp8(q_r, k, v, kv_len - n, out=o_r, km=km, scales=scales)
            else:
                sp.forward_sm120(q_r, k, v, kv_len - n, out=o_r, km=km)
            continue
        k = kc[:, pages].reshape(Hk, npages * page, D)   # one gather copy, head-major
        v = vc[:, pages].reshape(Hk, npages * page, D)
        if npages * page > kv_len:
            k[:, kv_len:] = 0                            # keep block means clean; masked by causality anyway
        sp.forward(q_r, k, v, kv_len - n, out=o_r)
    return True


def _dense_request(impl, query, key_cache, value_cache, output, bt_row, n, kv_len):
    """Dense paged FlashAttention for one request of a mixed batch (same call as the backend)."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    dev = query.device
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    su = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    window = getattr(impl, 'sliding_window', None)
    flash_attn_varlen_func(q=query, k=key_cache, v=value_cache, out=output, cu_seqlens_q=cu, max_seqlen_q=n,
                           seqused_k=su, max_seqlen_k=kv_len, softmax_scale=impl.scale, causal=True,
                           alibi_slopes=getattr(impl, 'alibi_slopes', None),
                           window_size=list(window) if window is not None else None,
                           block_table=bt_row.unsqueeze(0), softcap=getattr(impl, 'logits_soft_cap', 0.0),
                           fa_version=getattr(impl, 'vllm_flash_attn_version', 2))
