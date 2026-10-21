"""Qwen dense BF16 MLP with a persistent gate/up+SwiGLU prefill stage.

Installed next to qwen2_moe.py by prepare_fused_task.py. Small and partial
tiles use ordinary BF16 linear plus an activation for the packed row layout.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.logger import init_logger
from .qwen2_moe import Qwen2MoeMLP


logger = init_logger(__name__)
TILE = (256, 128, 32)
_COMPILED = {}


@triton.jit
def _packed_activation(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // N, index % N
    source_col = (col // 16) * 32 + col % 16
    gate = tl.load(X + row * (2 * N) + source_col, row < M, 0).to(tl.float32)
    up = tl.load(X + row * (2 * N) + source_col + 16, row < M, 0).to(tl.float32)
    value = gate * tl.sigmoid(gate) * up
    tl.store(Y + index, value, row < M)


@torch.library.custom_op('kai_qwen_long::gate_up_swiglu', mutates_args=())
def gate_up_swiglu(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = packed_weight.shape[0] // 2
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if (x.dtype == torch.bfloat16 and x.is_contiguous() and packed_weight.is_contiguous()
            # Long prefill is the target. At M=256 cuBLAS's projection rounding
            # differs enough to fail the unchanged numerical gate; retain its
            # linear path for small batches as well as partial tiles.
            and k == 5120 and n == 17408 and m >= 2048 and m % TILE[0] == 0):
        os.environ.setdefault('CUTE_DSL_ARCH', 'sm_120a')
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        import cuda.bindings.driver as cuda
        from .kai_swiglu_vendor import Sm120GemmKernel

        tensors = [from_dlpack(t.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
                   for t in (x, packed_weight, out)]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = (m, n, k, torch.cuda.current_device())
        if key not in _COMPILED:
            _COMPILED[key] = cute.compile(Sm120GemmKernel(cutlass.Float32, TILE, 2),
                                          *tensors, 188, stream)
            logger.info('KAI persistent BF16 gate/up+SwiGLU compiled: M=%d tile=%s', m, TILE)
        _COMPILED[key](*tensors, stream)
    else:
        gate_up = F.linear(x, packed_weight)
        _packed_activation[(triton.cdiv(m * n, 256),)](gate_up, out, m, n, 256)
    return out


@gate_up_swiglu.register_fake
def _gate_up_swiglu_fake(x, packed_weight):
    return x.new_empty((x.shape[0], packed_weight.shape[0] // 2))


class QwenPersistentMLP(Qwen2MoeMLP):
    def prepare_persistent_weights(self):
        if getattr(self, '_kai_packed_swiglu', False):
            return
        weight = self.gate_up_proj.weight
        if (weight.dtype != torch.bfloat16 or tuple(weight.shape) != (34816, 5120)
                or self.expert_gate is not None):
            raise ValueError('This prototype requires the fixed dense BF16 Qwen MLP')
        # Replace the storage once, before profiling/graph capture. Retain only
        # one weight copy; never repack or change the checkpoint during timing.
        weight.data = weight.data.view(2, 17408 // 16, 16, 5120).permute(
            1, 0, 2, 3).contiguous().view(34816, 5120)
        self._kai_packed_swiglu = True

    def forward(self, x):
        if not getattr(self, '_kai_packed_swiglu', False):
            return super().forward(x)
        activated = gate_up_swiglu(x, self.gate_up_proj.weight)
        out, _ = self.down_proj(activated)
        return out
