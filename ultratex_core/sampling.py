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

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .render import GRID_ORDER

log = logging.getLogger("UltraTex")


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

    def reference_ids(self, target_hw, ref_hws):
        """FLUX.2: t-axis planes 10, 20 ... ; FLUX.1 (UNO pe='d'): diagonal h/w offsets."""
        out = []
        sh, sw = target_hw
        for i, (h, w) in enumerate(ref_hws):
            if self.family == "flux2":
                out.append(self.ids(h, w, t=10.0 * (i + 1)))
            else:
                out.append(self.ids(h, w, h_off=sh, w_off=sw))
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
    memory_factor: float = 28.0,
):
    import comfy.model_management as mm
    import comfy.utils

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
    ref_ids = bb.reference_ids((th, tw), [(th, tw), (rh, rw)])
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

    # ---- load DiT
    total_tokens = x_ids.shape[1] + ctx.shape[1]
    dtype = compute_dtype(model)
    batch = 2 if use_cfg else 1
    mem = int(total_tokens * bb.hidden * mm.dtype_size(dtype) * memory_factor * batch)
    mm.load_models_gpu([model], memory_required=mem)
    device = mm.get_torch_device()
    log.info(
        "UltraTex %s: %d/%d target tokens, %d reference tokens, %d text tokens, %s, cfg=%s",
        family, n_fg, th * tw, refs.shape[1], ctx.shape[1], dtype, use_cfg,
    )

    x = x.to(device)
    tgt_idx_d = tgt_idx.to(device)
    refs_d = refs.to(device, dtype)
    ids_d = x_ids.to(device).expand(batch, -1, -1)
    ctx_d = ctx.to(device, dtype)
    txt_ids_d = txt_ids.to(device)
    y_d = y.to(device, dtype).expand(batch, -1) if (family == "flux1" and y is not None) else None
    guid_d = torch.full((batch,), guidance, device=device, dtype=dtype) if family == "flux1" else None
    transformer_options = model.model_options.get("transformer_options", {}).copy()

    sched_len = ((R * 3) // 8) * ((R * 2) // 8) // (16 * 16)
    timesteps = uno_schedule(steps, sched_len)
    pbar = comfy.utils.ProgressBar(steps)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        mm.throw_exception_if_processing_interrupted()
        x_fg = x[:, tgt_idx_d]
        inp = torch.cat([x_fg.to(dtype), refs_d], dim=1).expand(batch, -1, -1)
        t_vec = torch.full((batch,), t_curr, device=device, dtype=dtype)
        pred = dm.forward_orig(
            inp, ids_d, ctx_d, txt_ids_d, t_vec, y_d, guidance=guid_d, transformer_options=transformer_options
        )[:, :n_fg].float()
        if use_cfg:
            pred = pred[0:1] + guidance * (pred[1:2] - pred[0:1])
        x[:, tgt_idx_d] = x_fg + (t_prev - t_curr) * pred
        pbar.update(1)

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
