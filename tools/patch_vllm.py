"""Apply the sm120_attn hook (and optionally the BF16 MLP fusion) to a vLLM v0.30.0 source tree.

    python tools/patch_vllm.py --target $(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    python tools/patch_vllm.py --target vllm/vllm            # the git submodule, before building from source
    python tools/patch_vllm.py --target ... --with-mlp-fusion
    python tools/patch_vllm.py --target ... --check          # report without changing anything

The hook is inserted by anchor matching (idempotent), not by `patch`, so it also applies to an installed wheel.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANCHOR = '''            else:
                causal = attn_metadata.causal
                is_dynamic_causal = isinstance(causal, torch.Tensor)
'''
HOOK = '''            else:
                # SM120 dynamic block-sparse attention for long prefill chunks (sm120_attn)
                from sm120_attn.sparse_prefill import try_sparse_forward
                if try_sparse_forward(self, query[:num_actual_tokens], key[:num_actual_tokens], key_cache,
                                      value_cache, output[:num_actual_tokens], attn_metadata):
                    return output
                causal = attn_metadata.causal
                is_dynamic_causal = isinstance(causal, torch.Tensor)
'''
MLP_OLD = 'from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP\n'
MLP_NEW = 'from .kai_persistent_mlp import QwenPersistentMLP as Qwen3NextMLP\n'
MLP_CLASS = 'class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):\n'
MLP_METHOD = '''class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    def process_weights_after_loading(self):
        count = 0
        for module in self.language_model.modules():
            if isinstance(module, Qwen3NextMLP):
                module.prepare_persistent_weights()
                count += 1
        if count != 64:
            raise ValueError(f"Expected 64 dense Qwen MLPs, found {count}")
        logger.info("KAI prepared persistent BF16 SwiGLU weights for %d MLPs", count)

'''


def patch_file(path: Path, old: str, new: str, check: bool) -> str:
    text = path.read_text()
    if new in text:
        return 'already patched'
    if old not in text:
        return 'ANCHOR NOT FOUND (vLLM version mismatch?)'
    if not check:
        path.write_text(text.replace(old, new, 1))
    return 'patched' if not check else 'would patch'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=Path, required=True, help='directory of the vllm package (contains v1/, model_executor/)')
    ap.add_argument('--with-mlp-fusion', action='store_true', help='also install extras/bf16_mlp_fusion (BF16 weights only)')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    t = args.target.resolve()
    fa = t / 'v1/attention/backends/flash_attn.py'
    if not fa.exists():
        sys.exit(f'{fa} not found: --target must be the vllm package directory')
    print('flash_attn.py:', patch_file(fa, ANCHOR, HOOK, args.check))
    if args.with_mlp_fusion:
        models = t / 'model_executor/models'
        q = models / 'qwen3_5.py'
        print('qwen3_5.py import:', patch_file(q, MLP_OLD, MLP_NEW, args.check))
        print('qwen3_5.py method:', patch_file(q, MLP_CLASS, MLP_METHOD, args.check))
        for f in ('kai_persistent_mlp.py', 'kai_swiglu_vendor.py'):
            src = HERE / 'extras/bf16_mlp_fusion' / f
            if not args.check:
                shutil.copy(src, models / f)
            print(f'{f}: {"would copy" if args.check else "copied"} -> {models / f}')
    try:
        import sm120_attn  # noqa: F401
        print('sm120_attn importable:', Path(sm120_attn.__file__).parent)
    except ImportError:
        print('NOTE: sm120_attn is not importable in this environment; run `pip install -e .` in the repository root')


if __name__ == '__main__':
    main()
