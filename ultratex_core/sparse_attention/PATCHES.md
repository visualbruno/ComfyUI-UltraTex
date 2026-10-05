# Vendored SLA with patches

This directory vendors [SLA (Sparse Linear Attention)](https://github.com/thu-ml/SLA)
by Jintao Zhang, Haoxu Wang et al., from the `sparse_linear_attention/` package at
upstream commit [`552a51f`](https://github.com/thu-ml/SLA/commit/552a51f) (2025-12-21).
The upstream Apache-2.0 license is preserved unchanged in `LICENSE.txt`.

We vendor rather than depend on the upstream package because UltraTex needs the
changes below. Running against unpatched upstream SLA gives different numerics,
and on H100 with Triton 3.1 it fails outright.

## Changes

### 1. `core.py` — sparse branch only (linear branch removed)

Upstream `SparseLinearAttention` adds a linear-attention branch to the sparse
output: `o = o_s + proj_l(linear_attention(φ(q), φ(k), v))`, where `proj_l` is a
zero-initialised `nn.Linear(head_dim, head_dim)`. UltraTex drops that branch and
returns the sparse block attention alone, `o = o_s`. `proj_l`, `init_weights_` and
the linear-attention computation are removed, so the module has no trainable
parameters and drops into a pretrained FLUX transformer without new weights.
The `feature_map` argument is still accepted for interface compatibility but is
unused in the forward pass.

### 2. `kernel.py` / `utils.py` — masked loads need `other=0.0`

A masked `tl.load` without `other` leaves masked lanes undefined. Those lanes
take part in the block reductions that follow, so a partial final block leaks
garbage into the output and its gradients. `other=0.0` is added at all 16 masked
load sites (forward, backward and the block-mean pre-pass).

### 3. `kernel.py` — unconditional tail masking

`idx_n` is loaded dynamically from the LUT. A runtime `if` on this value
triggers an invalid instruction schedule in Triton 3.1 on H100
("operation scheduled before its operands"). Applying the mask unconditionally
is equivalent: it is all-true for complete blocks and only changes the padded
tail of the final incomplete block.

```python
# before
if L - idx_n * BLOCK_N < BLOCK_N:
    qk = tl.where(n_mask[None, :], qk, float("-inf"))
# after
qk = tl.where(n_mask[None, :], qk, float("-inf"))
```

### 4. `utils.py` — explicit fp32 reduction

Triton 3.x removed the `dtype=` reduction argument. Cast explicitly so
accumulation stays fp32, matching the original SLA implementation.

```python
x_mean = tl.sum(x.to(tl.float32), axis=0) / nx
```

`__init__.py` is unchanged from upstream.

## Configuration

UltraTex uses `head_dim=128, topk=0.20, BLKQ=128, BLKK=64` for both backbones.
See `ultratex/backbones/flux1/math.py` and
`ultratex/backbones/flux2/model.py:configure_sparse_attention`.
