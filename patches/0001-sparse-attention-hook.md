# Patch 0001: block-sparse prefill hook for the vLLM FlashAttention backend (vLLM v0.30.0)

`tools/patch_vllm.py` inserts the following lines into `vllm/v1/attention/backends/flash_attn.py`, in
`FlashAttentionImpl.forward`, right before the ordinary `flash_attn_varlen_func` call of the non-cascade path
(the `else:` branch that begins with `causal = attn_metadata.causal`):

```python
                # SM120 dynamic block-sparse attention for long prefill chunks (sm120_attn)
                from sm120_attn.sparse_prefill import try_sparse_forward
                if try_sparse_forward(self, query[:num_actual_tokens], key[:num_actual_tokens], key_cache,
                                      value_cache, output[:num_actual_tokens], attn_metadata):
                    return output
```

The hook returns False (and writes nothing) for batches without a qualifying long prefill chunk, so decode
steps and short prefills take vLLM's normal path. Everything else lives in the `sm120_attn` package.
