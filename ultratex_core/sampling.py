"""UltraTex multi-view sampling on top of ComfyUI's Flux / Flux2 diffusion models.

Faithful to UltraTex's inference (train_flux1.inference / train_flux2.inference, "ai" dataset):
  * target = 2x3 atlas of the 6 views; references = 2x3 world-normal atlas + reference image
  * Background Token Dropping: only foreground tokens of target and references enter the DiT
    (masks at resolution/16, 2x2 dilated), original RoPE positions kept
  * FLUX.2: reference ids on the t axis (10, 20), text CFG only when a negative is given
  * FLUX.1: reference ids offset diagonally ('d'), distilled guidance embedding
  * UNO time-shifted schedule, Euler steps, background latent = encoded black image
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .render import GRID_ORDER

log = logging.getLogger("UltraTex")

FLUX1_TRAIN_RES = 2048  # resolution the UltraTex FLUX.1 LoRA was trained at (train_flux1.py)


# --------------------------------------------------------------------------- conditions
def _resize_rgba(rgba: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(rgba).resize((size, size), Image.Resampling.BILINEAR)


def _on_white(img: Image.Image) -> np.ndarray:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.getchannel("A"))
    return np.asarray(bg)


def _small_mask(rgba: np.ndarray, size: int) -> np.ndarray:
    """alpha > 0 at token resolution, dilated like cv2.dilate(mask, ones((2, 2))) in UltraTex."""
    m = np.asarray(_resize_rgba(rgba, size).getchannel("A")) > 0
    d = m.copy()
    d[1:] |= m[:-1]
    d[:, 1:] |= m[:, :-1]
    d[1:, 1:] |= m[:-1, :-1]
    return d


def _grid(tiles: list) -> np.ndarray:
    return np.concatenate([np.concatenate([tiles[v] for v in row], axis=1) for row in GRID_ORDER], axis=0)


def build_conditions(prep, resolution: int) -> dict:
    """Images/masks exactly as FluxPairedDatasetAI builds them (target alpha == normal alpha)."""
    R, s = resolution, resolution // 16
    views = prep.normal_rgba
    normal_tiles, masks_small, masks_full = [], [], []
    for v in range(6):
        big = _resize_rgba(views[v], R)
        normal_tiles.append(_on_white(big))
        masks_full.append(np.asarray(big.getchannel("A")) > 0)
        masks_small.append(_small_mask(views[v], s))
    ref_big = _resize_rgba(prep.reference_rgba, R)
    return {
        "normal_tiles": np.stack(normal_tiles),  # (6, R, R, 3) uint8, view order
        "reference": _on_white(ref_big)[None],  # (1, R, R, 3) uint8
        "mask_target": _grid(masks_small).astype(bool),  # (2s, 3s)
        "mask_normals": _grid(masks_small).astype(bool),
        "mask_reference": _small_mask(prep.reference_rgba, s).astype(bool),  # (s, s)
        "masks_full": np.stack(masks_full),  # (6, R, R) view order
    }


# --------------------------------------------------------------------------- schedule
def uno_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    slope = (max_shift - base_shift) / (4096 - 256)
    mu = slope * image_seq_len + (base_shift - slope * 256)
    return (math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)).tolist()


# --------------------------------------------------------------------------- latents
def _to_pixels(arr_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_uint8.astype(np.float32) / 255.0)


def encode_tiles(vae, model, tiles_uint8: np.ndarray) -> torch.Tensor:
    """VAE-encode (N, R, R, 3) tiles one by one (as UltraTex does), in the model's latent space."""
    out = [model.model.process_latent_in(vae.encode(_to_pixels(t[None]))) for t in tiles_uint8]
    return torch.cat(out, dim=0).float()


def tiles_to_grid(lat: torch.Tensor) -> torch.Tensor:
    """(6, C, h, w) in view order -> (1, C, 2h, 3w) atlas."""
    rows = [torch.cat([lat[v] for v in row], dim=-1) for row in GRID_ORDER]
    return torch.cat(rows, dim=-2)[None]


def grid_to_tiles(grid: torch.Tensor) -> torch.Tensor:
    """(1, C, 2h, 3w) atlas -> (6, C, h, w) in view order."""
    h, w = grid.shape[-2] // 2, grid.shape[-1] // 3
    out = [None] * 6
    for r, row in enumerate(GRID_ORDER):
        for c, v in enumerate(row):
            out[v] = grid[0, :, r * h : (r + 1) * h, c * w : (c + 1) * w]
    return torch.stack(out)


class Backbone:
    """Token packing / position ids for one Flux family."""

    def __init__(self, dm, family: str):
        self.family = family
        self.patch = dm.patch_size
        self.axes = len(dm.params.axes_dim)
        self.hidden = dm.hidden_size

    def pack(self, lat: torch.Tensor) -> torch.Tensor:
        p = self.patch
        return rearrange(lat, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p)

    def unpack(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        p = self.patch
        return rearrange(tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=p, pw=p)

    def ids(self, h: int, w: int, t: float = 0.0, h_off: int = 0, w_off: int = 0) -> torch.Tensor:
        ids = torch.zeros(h, w, self.axes)
        ids[..., 0] = t
        ids[..., 1] = torch.arange(h)[:, None] + h_off
        ids[..., 2] = torch.arange(w)[None, :] + w_off
        return ids.reshape(1, h * w, self.axes)

    def reference_ids(self, target_hw, ref_hws, offset_scale: float = 1.0):
        """FLUX.2: t-axis planes 10, 20 ... ; FLUX.1 (UNO pe='d'): diagonal h/w offsets, scaled by
        `offset_scale` (token spacing itself is unchanged)."""
        out = []
        sh, sw = target_hw
        for i, (h, w) in enumerate(ref_hws):
            if self.family == "flux2":
                out.append(self.ids(h, w, t=10.0 * (i + 1)))
            else:
                out.append(self.ids(h, w, h_off=sh * offset_scale, w_off=sw * offset_scale))
                sh, sw = sh + h, sw + w
        return out

    def txt_ids(self, length: int) -> torch.Tensor:
        ids = torch.zeros(1, length, self.axes)
        if self.family == "flux2":
            ids[..., 3] = torch.arange(length)
        return ids


def model_family(model) -> str:
    import comfy.model_base

    if isinstance(model.model, comfy.model_base.Flux2):
        return "flux2"
    if isinstance(model.model, comfy.model_base.Flux):
        return "flux1"
    raise TypeError(f"UltraTex needs a FLUX.1 or FLUX.2 model, got {type(model.model).__name__}")


def compute_dtype(model) -> torch.dtype:
    dtype = getattr(model.model, "manual_cast_dtype", None) or model.model.get_dtype()
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        dtype = torch.bfloat16
    return dtype


# --------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample(
    model,
    vae,
    positive,
    negative,
    prep,
    resolution: int,
    steps: int,
    seed: int,
    guidance: float,
    drop_background_tokens: bool = True,
    memory_factor: float = 0.0,
    attention: str = "ultratex_sparse",
    sparse_topk: float = 0.2,
    token_chunk: int = 8192,
    init_views: np.ndarray | None = None,
    denoise: float = 1.0,
    keep_mask: np.ndarray | None = None,
):
    """init_views (6, H, W, 3) uint8 + denoise < 1: start from these views (SDEdit) instead of noise.
    keep_mask (6, H, W) float in [0, 1]: tokens are pulled back to the (noised) init views after every
    step in proportion to the mask (RePaint), so kept regions end exactly on the init content."""
    import time

    import comfy.model_management as mm
    import comfy.utils

    from . import optim

    family = model_family(model)
    dm = model.model.diffusion_model
    bb = Backbone(dm, family)
    R = resolution
    cond = build_conditions(prep, R)

    # ---- encode conditions (VAE)
    normal_lat = tiles_to_grid(encode_tiles(vae, model, cond["normal_tiles"]))
    ref_lat = encode_tiles(vae, model, cond["reference"])
    bg_lat = encode_tiles(vae, model, np.zeros((1, R, R, 3), np.uint8)).repeat(1, 1, 2, 3)
    th, tw = normal_lat.shape[-2] // bb.patch, normal_lat.shape[-1] // bb.patch  # token grid of the atlas
    rh, rw = ref_lat.shape[-2] // bb.patch, ref_lat.shape[-1] // bb.patch
    if (th, tw) != cond["mask_target"].shape or (rh, rw) != cond["mask_reference"].shape:
        raise ValueError(f"token grid {(th, tw)} / {(rh, rw)} does not match masks {cond['mask_target'].shape}")

    target_ids = bb.ids(th, tw)
    # FLUX.1 places references at h/w offsets equal to the grid sizes, which the LoRA learned at 2048:
    # keep those 2048 offsets at any resolution (at 1024 the native offsets made the model lose the
    # target <-> normal correspondence, e.g. a face painted on the back view). FLUX.2 uses t-planes.
    ref_ids = bb.reference_ids((th, tw), [(th, tw), (rh, rw)], offset_scale=FLUX1_TRAIN_RES / R)
    ref_tokens = [bb.pack(normal_lat), bb.pack(ref_lat)]
    bg_tokens = bb.pack(bg_lat)

    if drop_background_tokens:
        tgt_idx = torch.from_numpy(np.flatnonzero(cond["mask_target"]))
        ref_idx = [torch.from_numpy(np.flatnonzero(cond["mask_normals"])), torch.from_numpy(np.flatnonzero(cond["mask_reference"]))]
    else:
        tgt_idx = torch.arange(th * tw)
        ref_idx = [torch.arange(t.shape[1]) for t in ref_tokens]
    n_fg = len(tgt_idx)
    refs = torch.cat([t[:, i] for t, i in zip(ref_tokens, ref_idx)], dim=1)
    refs_ids = torch.cat([t[:, i] for t, i in zip(ref_ids, ref_idx)], dim=1)
    x_ids = torch.cat([target_ids[:, tgt_idx], refs_ids], dim=1)

    # ---- text
    ctx = positive[0][0]
    y = positive[0][1].get("pooled_output")
    use_cfg = family == "flux2" and negative is not None and guidance != 1.0
    if use_cfg and negative[0][0].shape == ctx.shape and torch.equal(negative[0][0], ctx):
        # uncond == cond (e.g. both prompts empty): u + g * (c - u) == c, so CFG is an exact no-op
        log.info("UltraTex: negative == positive, skipping CFG (identical result, half the compute)")
        use_cfg = False
    if use_cfg:
        neg = negative[0][0]
        if neg.shape[1] != ctx.shape[1]:
            raise ValueError("positive and negative conditioning must have the same token length")
        ctx = torch.cat([neg, ctx], dim=0)
    if family == "flux2" and ctx.shape[1] != 512:
        log.warning("UltraTex: FLUX.2 text context has %d tokens (UltraTex was trained with 512)", ctx.shape[1])
    txt_ids = bb.txt_ids(ctx.shape[1]).expand(ctx.shape[0], -1, -1)

    # ---- noise (CPU generator for reproducibility, like ComfyUI)
    gen = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn((1, th * tw, bg_tokens.shape[-1]), generator=gen, dtype=torch.float32)
    noise = x.clone()

    # ---- optional init views (SDEdit) and keep mask (RePaint)
    sched_len = ((R * 3) // 8) * ((R * 2) // 8) // (16 * 16)
    timesteps = uno_schedule(steps, sched_len)
    start = 0
    x0_init = keep_w = None
    if init_views is not None:
        tiles_init = np.stack([np.asarray(Image.fromarray(v).resize((R, R), Image.Resampling.BILINEAR)) for v in init_views])
        x0_init = bb.pack(tiles_to_grid(encode_tiles(vae, model, tiles_init)))
        if denoise < 1.0:
            start = next((i for i, t in enumerate(timesteps[:-1]) if t <= denoise), len(timesteps) - 2)
            t0 = timesteps[start]
            x = (1 - t0) * x0_init + t0 * noise
        if keep_mask is not None:
            s = R // 16
            km = [np.asarray(Image.fromarray(np.clip(m * 255, 0, 255).astype(np.uint8)).resize((s, s), Image.Resampling.BOX)) / 255.0
                  for m in keep_mask]
            keep_w = torch.from_numpy(_grid(km).reshape(-1)).float()[None, :, None]
        log.info("UltraTex: init views, denoise %.2f -> %d of %d steps%s", denoise, steps - start, steps,
                 f", keeping {float(keep_w[0, tgt_idx, 0].mean()):.0%} of the foreground" if keep_w is not None else "")
    elif denoise < 1.0 or keep_mask is not None:
        raise ValueError("UltraTex: denoise < 1 and keep_mask need init_views")

    # ---- load DiT
    total_tokens = x_ids.shape[1] + ctx.shape[1]
    dtype = compute_dtype(model)
    batch = 2 if use_cfg else 1
    if memory_factor <= 0:  # auto: peak activations per token, measured (x hidden x dtype size)
        memory_factor = 12.0 if token_chunk > 0 else 28.0
    mem = int(total_tokens * bb.hidden * mm.dtype_size(dtype) * memory_factor * batch)
    mm.load_models_gpu([model], memory_required=mem)
    device = mm.get_torch_device()

    transformer_options = model.model_options.get("transformer_options", {}).copy()
    if attention == "ultratex_sparse":
        err = optim.check_sparse_attention()
        if err is None:
            transformer_options["optimized_attention_override"] = optim.sparse_attention_override(sparse_topk)
        else:
            log.warning("UltraTex: sparse attention unavailable (%s), using ComfyUI attention", err)
            attention = "comfy_default"
    if attention != "ultratex_sparse" and optim.PROF.enabled:
        transformer_options["optimized_attention_override"] = optim.timing_attention_override()
    log.info(
        "UltraTex %s: %d/%d target tokens, %d reference tokens, %d text tokens, %s, cfg=%s, attention=%s%s, chunk=%s, reserve=%.1f GB",
        family, n_fg, th * tw, refs.shape[1], ctx.shape[1], dtype, use_cfg, attention,
        f"(top-k {sparse_topk})" if attention == "ultratex_sparse" else "", token_chunk or "off", mem / 2**30,
    )

    x = x.to(device)
    tgt_idx_d = tgt_idx.to(device)
    refs_d = refs.to(device, dtype)
    ids_d = x_ids.to(device).expand(batch, -1, -1)
    ctx_d = ctx.to(device, dtype)
    txt_ids_d = txt_ids.to(device)
    y_d = y.to(device, dtype).expand(batch, -1) if (family == "flux1" and y is not None) else None
    guid_d = torch.full((batch,), guidance, device=device, dtype=dtype) if family == "flux1" else None
    if family == "flux1" and ctx.shape[1] != 512:
        log.warning("UltraTex: FLUX.1 T5 context has %d tokens (UltraTex uses 512); use 'UltraTex Text Encode'", ctx.shape[1])

    keep_d = None
    if keep_w is not None:
        x0_d, noise_d = x0_init.to(device), noise.to(device)
        keep_d = keep_w.to(device)[:, tgt_idx_d]
    pbar = comfy.utils.ProgressBar(steps)
    pbar.update(start)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    with optim.chunked_model(dm, transformer_options, token_chunk) as topts:
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            if i < start:
                continue
            mm.throw_exception_if_processing_interrupted()
            t_step = time.time()
            x_fg = x[:, tgt_idx_d]
            inp = torch.cat([x_fg.to(dtype), refs_d], dim=1).expand(batch, -1, -1)
            t_vec = torch.full((batch,), t_curr, device=device, dtype=dtype)
            pred = dm.forward_orig(
                inp, ids_d, ctx_d, txt_ids_d, t_vec, y_d, guidance=guid_d, transformer_options=topts
            )[:, :n_fg].float()
            if use_cfg:
                pred = pred[0:1] + guidance * (pred[1:2] - pred[0:1])
            if i == start and os.environ.get("ULTRATEX_DUMP"):  # debug: inputs + first prediction for A/B checks
                torch.save({
                    "family": family, "x": x.cpu(), "tgt_idx": tgt_idx, "target_ids": target_ids,
                    "ref_tokens": ref_tokens, "ref_ids": ref_ids, "ref_idx": ref_idx,
                    "masks": [cond["mask_target"], cond["mask_normals"], cond["mask_reference"]],
                    "ctx": ctx.cpu(), "txt_ids": txt_ids.cpu(), "y": None if y is None else y.cpu(),
                    "t": t_curr, "guidance": guidance, "pred": pred.cpu(),
                }, os.environ["ULTRATEX_DUMP"])
                log.info("UltraTex: dumped first-step inputs/prediction to %s", os.environ["ULTRATEX_DUMP"])
            x_new = x_fg + (t_prev - t_curr) * pred
            if keep_d is not None:  # RePaint: kept tokens follow the init content's own noising path
                known = (1 - t_prev) * x0_d[:, tgt_idx_d] + t_prev * noise_d[:, tgt_idx_d]
                x_new = keep_d * known + (1 - keep_d) * x_new
            x[:, tgt_idx_d] = x_new
            pbar.update(1)
            optim.PROF.report(i + 1, time.time() - t_step)
            if i == start:
                log.info("UltraTex: first step %.1fs", time.time() - t_start)
    ran = max(steps - start, 1)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    log.info("UltraTex: %d steps in %.0fs (%.1fs/step), peak VRAM %.1f GB", ran, time.time() - t_start,
             (time.time() - t_start) / ran, peak)

    # ---- decode: foreground into the black-background latent, then tile by tile
    out_tokens = bg_tokens.to(device).clone()
    out_tokens[:, tgt_idx_d] = x[:, tgt_idx_d].to(out_tokens.dtype)
    grid = bb.unpack(out_tokens.cpu(), th, tw)
    tiles = grid_to_tiles(grid)
    images = vae.decode(model.model.process_latent_out(tiles))
    images = images.reshape(-1, *images.shape[-3:])[:6].float().cpu()
    masks = torch.from_numpy(cond["masks_full"]).float()
    images = images * masks[..., None] + (1 - masks[..., None])  # white background, as UltraTex
    return images, masks
