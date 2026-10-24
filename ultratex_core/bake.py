"""Back-project the generated views into a UV texture and build a textured trimesh.

Texels are rasterised in UV space; for every texel and view a quality q = cos(normal, view dir) x
ramp(distance to silhouette / occlusion edges) is computed where the texel passes the depth test.

fusion = "two_band" (default)
  * every view is split into a low band (Gaussian blur, sigma ~4 px at 1024) and a detail band
  * low band: smooth weighted blend, weights q^power        -> no exposure / colour seams
  * detail band: weights (q / q_max)^detail_power            -> detail from (nearly) the best view only,
    with a continuous hand-over instead of an argmax switch  -> no ghosting, no hard seam lines
  * per-view colour gains (robust log-domain least squares on overlaps, +-15 %, front view anchored)
  * second-pass views compete with a reduced quality (extra_view_bias): they paint a texel only where they
    see it clearly better than the first pass
  * fusion in linear light for colour; metallic-roughness is fused as plain data
  * per-view work (background fill, low band, visibility, bicubic sampling) runs in torch on the GPU
  * texels seen by no view (and, partially, weakly seen ones) are filled by a harmonic interpolation
    over the welded mesh surface (follows connectivity, continues across UV seams)
fusion = "legacy"
  * the original blend: cos^power weights, 50 % weighted mean + 50 % best view, nearest-3D hole fill

Chart padding is filled by UV-space nearest-neighbour dilation in both modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from .render import FOCAL_OVER_SIZE, GLRenderer, orbit_pose, project, rig_for_view, view_matrix

log = logging.getLogger("UltraTex")


@dataclass
class BakeSettings:
    power: float = 4.0  # legacy: cos^power blend weight; two_band: low-band blend exponent q^power
    min_cos: float = 0.05
    depth_eps: float = 0.004  # legacy fixed depth tolerance
    edge_jump: float = 0.01
    edge_px: float = 6.0  # edge feather, in px at 1024 (two_band scales it with the view size)
    best_view_mix: float = 0.5  # legacy only
    min_weight: float = 1e-4
    white_rim_px: float = 12.0  # at 1024 px views; near-white pixels this close inside a silhouette are dropped
    fusion: str = "two_band"
    detail_power: float = 32.0
    low_sigma_px: float = 4.0  # low/detail split, in px at 1024 (scaled with the view size)
    exposure_match: bool = True
    hole_fill: str = "mesh"  # "mesh" (harmonic over the surface) or "nearest" (4 nearest seen texels)
    max_gain: float = 0.15
    sampler: str = "bicubic"  # two_band detail-band lookup: "bicubic" or "bilinear"
    extra_view_bias: float = 0.6  # quality factor of second-pass views: they win a texel only when clearly better
    side_view_bias: float = 1.0  # same for the first pass' non-front views (< 1: the front view, aligned with the reference, wins near-ties)


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x0 + 1] * fx * (1 - fy)
        + img[y0 + 1, x0] * (1 - fx) * fy
        + img[y0 + 1, x0 + 1] * fx * fy
    )


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _drop_white_rim(view: np.ndarray, mask: np.ndarray, rim_px: float) -> np.ndarray:
    """Valid-pixel mask without the near-white rim the model sometimes leaves just inside silhouettes
    (generated outline a few pixels inside the true one, the gap left as white background)."""
    if rim_px <= 0:
        return mask
    rim_px = rim_px * view.shape[0] / 1024.0
    near_edge = ndimage.distance_transform_edt(mask) <= rim_px
    rim = mask & near_edge & (view > 0.9).all(-1)
    return mask & ~rim


def _extend_into_background(view: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy the nearest foreground colour into background pixels, so bilinear lookups at silhouettes never
    blend in the white background (that showed up as white fringes on sleeves / hands / boots)."""
    if mask.all() or not mask.any():
        return view
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return view[iy, ix]


# ---------------------------------------------------------------------- torch (GPU) helpers, two_band path
def _torch_device():
    import torch

    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _t_srgb_to_linear(c):
    import torch

    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _t_max_filter(x, r: int):
    """Square max filter of radius r on a (H, W) float map (separable)."""
    import torch.nn.functional as F

    k = 2 * r + 1
    x = F.max_pool2d(x[None, None], (1, k), 1, (0, r))
    return F.max_pool2d(x, (k, 1), 1, (r, 0))[0, 0]


def _t_edge_distance(edges, radius: float):
    """min(distance to the nearest edge pixel, radius): alternating 8- / 4-neighbour dilations (octagonal
    metric, within ~8 % of Euclidean). Only distances up to the feather radius are needed."""
    import torch
    import torch.nn.functional as F

    reached = edges.float()[None, None]
    dist = torch.zeros_like(reached)
    for i in range(int(np.ceil(radius))):
        dist += 1 - reached
        if i % 2 == 0:
            reached = F.max_pool2d(reached, 3, 1, 1)
        else:
            reached = torch.maximum(F.max_pool2d(reached, (1, 3), 1, (0, 1)), F.max_pool2d(reached, (3, 1), 1, (1, 0)))
    return dist[0, 0]


def _t_fill_background(img, valid):
    """(1, C, H, W) image with ~valid pixels filled from nearby valid ones (pull-push pyramid); valid
    pixels are unchanged. Same purpose as _extend_into_background, without a full-size distance transform."""
    import torch
    import torch.nn.functional as F

    c, w = img * valid, valid.float()
    pyramid = []
    while min(c.shape[-2:]) > 1:
        pyramid.append((c, w))
        c, w = F.avg_pool2d(c, 2, ceil_mode=True), F.avg_pool2d(w, 2, ceil_mode=True)
    fill = c / w.clamp_min(1e-8)
    for c, w in reversed(pyramid):
        up = F.interpolate(fill, size=c.shape[-2:], mode="bilinear", align_corners=False)
        fill = torch.where(w > 0, c / w.clamp_min(1e-8), up)
    return fill


def _t_gaussian(x, sigma: float):
    """Separable Gaussian blur of a (1, C, H, W) tensor, edges replicated."""
    import torch
    import torch.nn.functional as F

    r = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    ch = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(ch, 1, 1, -1), groups=ch)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(ch, 1, -1, 1), groups=ch)


class Baker:
    def __init__(self, prep, tex_size: int):
        self.prep = prep
        self.tex_size = tex_size
        self.renderer = GLRenderer(prep.vertices, prep.normals, prep.faces, prep.uvs)
        tpos, tnrm, tdep = self.renderer.render_uv(tex_size)
        face = self.renderer.last_face
        self.cover = tdep[..., 3] > 0
        self.texel_pos = tpos[..., :3][self.cover].astype(np.float64)
        n = np.nan_to_num(tnrm[..., :3][self.cover].astype(np.float64))  # zero normal => weight 0
        self.texel_nrm = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        self.texel_face = np.clip(np.rint(face[self.cover]).astype(np.int64), 0, len(prep.faces) - 1)
        self._view_cache = {}
        self._surface = None
        self._texels_t = None
        log.info("UltraTex bake: UV coverage %.3f of %d^2 texels", self.cover.mean(), tex_size)

    def release(self):
        self.renderer.release()

    # ------------------------------------------------------------------ geometry per view
    def _view_geometry(self, view: int, size: int, mask: np.ndarray, s: BakeSettings, rig: np.ndarray | None):
        """legacy fusion: per-texel (u, v, weight, quality) for one view, fixed depth tolerance."""
        rig = rig_for_view(rig, view)
        key = (view, size, None if rig is None else rig.tobytes())
        if key not in self._view_cache:
            _, _, dep = self.renderer.render_view(view, size, rig)
            self._view_cache[key] = (dep[..., 0], dep[..., 3] > 0)
        depth, cover = self._view_cache[key]
        gy, gx = np.gradient(np.where(cover, depth, depth[cover].max() + 1.0))
        edges = (~cover) | (np.hypot(gx, gy) > s.edge_jump) | (~mask)
        edge_dist = ndimage.distance_transform_edt(~edges)

        u, v, d = project(self.texel_pos, view, size, rig)
        ui = np.clip(np.floor(u).astype(int), 0, size - 1)
        vi = np.clip(np.floor(v).astype(int), 0, size - 1)
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        cam = np.linalg.inv(view_matrix(view, rig))[:3, 3]  # camera centre in the object frame
        to_cam = cam[None] - self.texel_pos
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        cos = np.einsum("ij,ij->i", self.texel_nrm, to_cam)
        visible = inside & cover[vi, ui] & (np.abs(d - depth[vi, ui]) < s.depth_eps)
        ramp = np.clip(edge_dist[vi, ui] / s.edge_px, 0.0, 1.0)
        ok = visible & (cos > s.min_cos)
        w = np.where(ok, np.clip(cos, 0, 1) ** s.power * ramp, 0.0)
        quality = np.where(ok, np.clip(cos, 0, 1) * ramp, 0.0)  # how head-on (and away from edges)
        return u, v, w, quality

    def _view_geometry_t(self, view: int, size: int, valid, s: BakeSettings, rig: np.ndarray | None, dev):
        """two_band, on `dev`: (u, v, quality) per texel, same model as _view_geometry with view-size scaling."""
        import torch

        rig = rig_for_view(rig, view)
        key = (view, size, None if rig is None else rig.tobytes())
        if key not in self._view_cache:
            dep = self.renderer.render_depth(view, size, rig)
            self._view_cache[key] = (np.ascontiguousarray(dep[..., 0]), dep[..., 3] > 0)
        depth, cover = (torch.from_numpy(a).to(dev) for a in self._view_cache[key])
        if self._texels_t is None or self._texels_t[0] != dev:
            self._texels_t = (dev, torch.from_numpy(self.texel_pos).float().to(dev), torch.from_numpy(self.texel_nrm).float().to(dev))
        _, pos, nrm = self._texels_t
        scale = size / 1024.0
        gy, gx = torch.gradient(torch.where(cover, depth, depth[cover].max() + 1.0))
        edges = ~cover | (torch.hypot(gx, gy) * scale > s.edge_jump) | ~valid
        radius = s.edge_px * scale
        edge_dist = _t_edge_distance(edges, radius)
        del gx, gy, edges

        mv = view_matrix(view, rig)
        p = pos @ torch.from_numpy(mv[:3, :3].T).float().to(dev) + torch.from_numpy(mv[:3, 3]).float().to(dev)
        d = -p[:, 2]
        f = FOCAL_OVER_SIZE * size
        u = size / 2 + f * p[:, 0] / d
        v = size / 2 - f * p[:, 1] / d
        del p
        ui = u.floor().long().clamp(0, size - 1)
        vi = v.floor().long().clamp(0, size - 1)
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        cam = torch.from_numpy(np.linalg.inv(mv)[:3, 3]).float().to(dev)
        to_cam = cam[None] - pos
        cos = (nrm * to_cam).sum(1) / to_cam.norm(dim=1).clamp_min(1e-8)
        del to_cam
        tol = 0.0015 + 2.0 * (d / f) / cos.abs().clamp_min(0.2)  # one pixel's world size, stretched at grazing angles
        flat = vi * size + ui
        visible = inside & cover.reshape(-1)[flat] & ((d - depth.reshape(-1)[flat]).abs() < tol)
        ramp = (edge_dist.reshape(-1)[flat] / radius).clamp(0.0, 1.0)
        ok = visible & (cos > s.min_cos)
        quality = torch.where(ok, cos.clamp(0.0, 1.0) * ramp, torch.zeros_like(cos))
        return u, v, quality

    def _sample_view(self, img: np.ndarray, mask: np.ndarray, rig, view: int, s: BakeSettings, data: bool, dev):
        """two_band: one view's observations -> (texel indices, quality, full sample, low-band sample), numpy."""
        import torch
        import torch.nn.functional as F

        size = img.shape[0]
        im = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).to(dev)
        valid = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
        rim_px = 0.0 if data else s.white_rim_px * size / 1024.0
        if rim_px > 0:  # near-white pixels close inside the silhouette (see _drop_white_rim)
            near_edge = _t_max_filter((~valid).float(), int(round(rim_px))) > 0
            valid = valid & ~(near_edge & (im > 0.9).all(-1))
            del near_edge
        u, v, q = self._view_geometry_t(view, size, valid, s, rig, dev)
        idx = torch.nonzero(q > 0).squeeze(1)
        if idx.numel() == 0:
            return None
        x = im.permute(2, 0, 1)[None]
        del im
        if not data:
            x = _t_srgb_to_linear(x)
        x = _t_fill_background(x, valid[None, None])
        factor = max(1, size // 1024)  # the low band is built at ~1024 px, then blurred
        low = F.interpolate(x, size=(size // factor, size // factor), mode="area") if factor > 1 else x
        low = _t_gaussian(low, s.low_sigma_px * size / 1024.0 / factor)
        grid = torch.stack((u[idx] / size * 2 - 1, v[idx] / size * 2 - 1), -1).view(1, 1, -1, 2)
        full = F.grid_sample(x, grid, mode=s.sampler, padding_mode="border", align_corners=False)[0, :, 0].T.clamp(0.0, 1.0)
        low = F.grid_sample(low, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T
        out = (idx.cpu().numpy(), q[idx].cpu().numpy(), full.cpu().numpy(), low.cpu().numpy())
        del x, low, full, grid, u, v, q, idx, valid
        return out

    @staticmethod
    def _iter_views(view_sets):
        for k, (views, masks, rig, *rest) in enumerate(view_sets):
            use = rest[0] if rest else None  # optional subset of the 6 views
            for view in range(6):
                if use is None or view in use:
                    yield views[view], masks[view], rig, view, k

    # ------------------------------------------------------------------ public
    def bake(self, view_sets, s: BakeSettings, data: bool = False):
        """view_sets: [(views (6, H, W, 3) float [0,1], masks (6, H, W) bool, rig or None[, view indices]), ...]
        data=True for scalar maps (metallic-roughness): no linear-light conversion, no colour gains.

        Returns (texture (T, T, 3) float, confidence (T, T) float = best per-texel view quality)."""
        if s.fusion == "legacy":
            colors, best_q = self._bake_legacy(view_sets, s)
        else:
            colors, best_q = self._bake_two_band(view_sets, s, data)
        tex = np.zeros((self.tex_size, self.tex_size, 3), np.float32)
        tex[self.cover] = colors
        conf = np.zeros((self.tex_size, self.tex_size), np.float32)
        conf[self.cover] = best_q
        _, (iy, ix) = ndimage.distance_transform_edt(~self.cover, return_indices=True)
        return tex[iy, ix], conf[iy, ix]

    # ------------------------------------------------------------------ legacy
    def _bake_legacy(self, view_sets, s: BakeSettings):
        n = len(self.texel_pos)
        acc = np.zeros((n, 3))
        wsum = np.zeros(n)
        best_w = np.zeros(n)
        best_c = np.zeros((n, 3))
        best_q = np.zeros(n)
        for img, mask, rig, view, _ in self._iter_views(view_sets):
            size = img.shape[0]
            valid = _drop_white_rim(img, mask, s.white_rim_px)
            u, v, w, q = self._view_geometry(view, size, valid, s, rig)
            col = bilinear(_extend_into_background(img, valid), u - 0.5, v - 0.5)
            acc += w[:, None] * col
            wsum += w
            better = w > best_w
            best_w[better], best_c[better] = w[better], col[better]
            np.maximum(best_q, q, out=best_q)
        seen = wsum > s.min_weight
        colors = np.zeros((n, 3))
        colors[seen] = acc[seen] / wsum[seen, None]
        colors[seen] = (1 - s.best_view_mix) * colors[seen] + s.best_view_mix * best_c[seen]
        log.info("UltraTex bake (legacy): %.3f of texels seen; filling the rest from nearest seen texels", seen.mean())
        if (~seen).any() and seen.any():
            _, idx = cKDTree(self.texel_pos[seen]).query(self.texel_pos[~seen], k=4)
            colors[~seen] = colors[seen][idx].mean(axis=1)
        return colors, best_q

    # ------------------------------------------------------------------ two-band
    def _bake_two_band(self, view_sets, s: BakeSettings, data: bool):
        n = len(self.texel_pos)
        dev = _torch_device()
        samples, biases = [], []  # per view: texel indices (each at most once), q, full sample, low-band sample
        for img, mask, rig, view, k in self._iter_views(view_sets):
            try:
                out = self._sample_view(img, mask, rig, view, s, data, dev)
            except Exception as e:  # out of GPU memory: finish on the CPU (same result, slower)
                if dev.type != "cuda" or "out of memory" not in str(e).lower():
                    raise
                import torch

                self._texels_t = None
                torch.cuda.empty_cache()
                dev = torch.device("cpu")
                log.warning("UltraTex bake: GPU out of memory, sampling the views on the CPU")
                out = self._sample_view(img, mask, rig, view, s, data, dev)
            if out is not None:
                samples.append(out)
                biases.append(s.extra_view_bias if k else 1.0 if view == 0 else s.side_view_bias)
        self._texels_t = None
        if dev.type == "cuda":
            import torch

            torch.cuda.empty_cache()
        qmax = np.zeros(n, np.float32)  # real best quality (trust / confidence)
        qmax_w = np.zeros(n, np.float32)  # best biased quality (detail hand-over)
        for (idx, q, _, _), b in zip(samples, biases):
            qmax[idx] = np.maximum(qmax[idx], q)
            qmax_w[idx] = np.maximum(qmax_w[idx], q * b)

        gains = self._colour_gains(samples, n, s) if (s.exposure_match and not data and len(samples) > 1) else None
        low_acc = np.zeros((n, 3)); low_w = np.zeros(n)
        det_acc = np.zeros((n, 3)); det_w = np.zeros(n)
        for k, ((idx, q, full, low), b) in enumerate(zip(samples, biases)):
            if gains is not None:
                full, low = full * gains[k], low * gains[k]
            q = q * b
            wl = q.astype(np.float64) ** s.power
            wd = (q / np.maximum(qmax_w[idx], 1e-6)).astype(np.float64) ** s.detail_power
            low_w[idx] += wl
            det_w[idx] += wd
            low_acc[idx] += wl[:, None] * low
            det_acc[idx] += wd[:, None] * (full - low)
        seen = low_w > s.min_weight
        colors = np.zeros((n, 3))
        colors[seen] = low_acc[seen] / low_w[seen, None] + det_acc[seen] / np.maximum(det_w[seen, None], 1e-12)
        colors = np.clip(colors, 0.0, 1.0)

        # weakly seen texels are blended into the surface fill, unseen ones take it entirely
        alpha = np.where(seen, _smoothstep(qmax, 0.05, 0.2), 0.0)
        if (alpha < 1).any() and seen.any():
            fill = self._mesh_fill(colors, alpha) if s.hole_fill == "mesh" else self._nearest_fill(colors, seen)
            colors = alpha[:, None] * colors + (1 - alpha[:, None]) * fill
        log.info("UltraTex bake (two-band%s): %.3f of texels seen, %.3f fully trusted, %d view(s)",
                 ", exposure-matched" if gains is not None else "", seen.mean(), (alpha >= 1).mean(), len(samples))
        if not data:
            colors = linear_to_srgb(colors)
        return colors, qmax

    def _colour_gains(self, samples, n: int, s: BakeSettings, max_texels: int = 200_000):
        """Per-view RGB gains from overlaps (log-domain robust least squares), first view anchored."""
        nv = len(samples)
        counts = np.zeros(n, np.int32)
        for idx, q, _, _ in samples:
            counts[idx[q > 0.3]] += 1
        cand = np.flatnonzero(counts >= 2)
        if len(cand) < 1000:
            return None
        rng = np.random.default_rng(0)
        sub = cand if len(cand) <= max_texels else rng.choice(cand, max_texels, replace=False)
        pos = np.full(n, -1, np.int64)
        pos[sub] = np.arange(len(sub))
        L = np.full((nv, len(sub), 3), np.nan, np.float32)
        Q = np.zeros((nv, len(sub)), np.float32)
        for k, (idx, q, _, low) in enumerate(samples):
            p = pos[idx]
            m = (p >= 0) & (q > 0.3)
            L[k, p[m]] = np.log(np.maximum(low[m], 1e-3))
            Q[k, p[m]] = q[m]
        x = np.zeros((nv, 3))
        for _ in range(4):  # IRLS with Cauchy weights
            H = np.zeros((3, nv, nv)); g = np.zeros((3, nv))
            for i in range(nv):
                for j in range(i + 1, nv):
                    both = (Q[i] > 0) & (Q[j] > 0)
                    if both.sum() < 200:
                        continue
                    r = (L[i, both] + x[i]) - (L[j, both] + x[j])  # (m, 3) current residual
                    base = (Q[i, both] * Q[j, both])[:, None]
                    w = base / (1.0 + (r / 0.1) ** 2)
                    d = L[i, both] - L[j, both]  # want x_i - x_j = -d
                    for c in range(3):
                        sw = w[:, c].sum()
                        H[c, i, i] += sw; H[c, j, j] += sw; H[c, i, j] -= sw; H[c, j, i] -= sw
                        g[c, i] -= (w[:, c] * d[:, c]).sum(); g[c, j] += (w[:, c] * d[:, c]).sum()
            scale = max(np.trace(H[0]) / nv, 1e-6)
            for c in range(3):
                Hc = H[c] + np.eye(nv) * 1e-2 * scale
                Hc[0, 0] += 1e3 * scale  # anchor the first view (front: aligned with the reference)
                x[:, c] = np.linalg.solve(Hc, g[c])
            x = np.clip(x, np.log(1 - s.max_gain), np.log(1 + s.max_gain))
        gains = np.exp(x).astype(np.float32)
        log.info("UltraTex bake: colour gains per view %s", np.round(gains.mean(1), 3).tolist())
        return gains

    # ------------------------------------------------------------------ hole filling
    def _surface_graph(self):
        if self._surface is None:
            prep = self.prep
            key = np.round(prep.vertices.astype(np.float64) * 1e5).astype(np.int64)
            _, wid = np.unique(key, axis=0, return_inverse=True)
            wid = wid.reshape(-1)
            nw = wid.max() + 1
            pos = np.zeros((nw, 3))
            np.add.at(pos, wid, prep.vertices)
            pos /= np.bincount(wid, minlength=nw)[:, None]
            fw = wid[prep.faces]
            e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]])
            e = e[e[:, 0] != e[:, 1]]
            e = np.unique(np.sort(e, axis=1), axis=0)
            wts = 1.0 / np.maximum(np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1), 1e-6)
            W = sparse.coo_matrix((np.r_[wts, wts], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nw, nw)).tocsr()
            # barycentric coordinates of every texel in its triangle
            tri = prep.vertices[prep.faces[self.texel_face]].astype(np.float64)
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            v0, v1, v2 = b - a, c - a, self.texel_pos - a
            d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
            d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
            den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-20, d00 * d11 - d01 * d01, 1.0)
            bv = (d11 * d20 - d01 * d21) / den
            bw = (d00 * d21 - d01 * d20) / den
            bary = np.clip(np.stack([1 - bv - bw, bv, bw], 1), 0, 1)
            bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
            self._surface = (fw[self.texel_face], bary, W, pos)
        return self._surface

    def _mesh_fill(self, colors: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Harmonic (Laplace) interpolation of the trusted texels over the welded mesh surface."""
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.linalg import cg, spsolve

        tv, bary, W, pos = self._surface_graph()
        nw = W.shape[0]
        wgt = bary * alpha[:, None]
        vw = np.bincount(tv.ravel(), weights=wgt.ravel(), minlength=nw)
        vc = np.stack([np.bincount(tv.ravel(), weights=(wgt * colors[:, c:c + 1]).ravel(), minlength=nw) for c in range(3)], 1)
        known = vw > 0.05
        vcol = np.zeros((nw, 3))
        vcol[known] = vc[known] / vw[known, None]
        unknown = ~known
        if unknown.any():
            # components without any known vertex: copy the nearest known vertex
            ncomp, comp = connected_components(W, directed=False)
            has_known = np.bincount(comp, weights=known.astype(float), minlength=ncomp) > 0
            orphan = unknown & ~has_known[comp]
            if orphan.any() and known.any():
                _, nn = cKDTree(pos[known]).query(pos[orphan])
                vcol[orphan] = vcol[known][nn]
                known = known | orphan
                unknown = ~known
        if unknown.any():
            Lap = sparse.diags(np.asarray(W.sum(1)).ravel()) - W
            U, K = np.flatnonzero(unknown), np.flatnonzero(known)
            A = Lap[U][:, U].tocsc()
            rhs = -(Lap[U][:, K] @ vcol[K])
            if len(U) <= 300_000:
                vcol[U] = spsolve(A, rhs).reshape(len(U), 3)
            else:
                diag = A.diagonal()
                pre = sparse.diags(1.0 / np.maximum(diag, 1e-12))
                for c in range(3):
                    vcol[U, c], _ = cg(A, rhs[:, c], M=pre, maxiter=2000)
            log.info("UltraTex bake: surface fill solved %d of %d vertices", len(U), nw)
        return np.clip((vcol[tv] * bary[..., None]).sum(1), 0.0, 1.0)

    def _nearest_fill(self, colors: np.ndarray, seen: np.ndarray) -> np.ndarray:
        fill = colors.copy()
        _, idx = cKDTree(self.texel_pos[seen]).query(self.texel_pos[~seen], k=4)
        fill[~seen] = colors[seen][idx].mean(axis=1)
        return fill

    # ------------------------------------------------------------------ rendering helpers
    def render_views(self, tex: np.ndarray, conf: np.ndarray | None, size: int, rig: np.ndarray | None):
        """Render the textured mesh from the 6 canonical cameras of a (rotated) rig.

        Returns views (6, size, size, 3) float on white, coverage (6, size, size) bool and the baked
        confidence seen through each pixel (6, size, size) float."""
        th, tw = tex.shape[:2]
        views, covers, confs = [], [], []
        for view in range(6):
            _, _, dep = self.renderer.render_view(view, size, rig_for_view(rig, view))
            cover = dep[..., 3] > 0
            x = (dep[..., 1] * tw - 0.5).ravel()
            y = ((1 - dep[..., 2]) * th - 0.5).ravel()
            col = bilinear(tex, x, y).reshape(size, size, 3)
            views.append(np.where(cover[..., None], col, 1.0))
            covers.append(cover)
            if conf is not None:
                c = bilinear(conf[..., None], x, y).reshape(size, size)
                confs.append(np.where(cover, c, 0.0))
        confs = np.stack(confs) if confs else np.zeros((6, size, size))
        return np.clip(np.stack(views), 0, 1).astype(np.float32), np.stack(covers), confs.astype(np.float32)

    def preview(self, tex: np.ndarray, size: int = 512, elevation: float = 15.0) -> np.ndarray:
        """Unlit renders at 8 azimuths -> (8, size, size, 3) float."""
        out = []
        th, tw = tex.shape[:2]
        for az in range(0, 360, 45):
            _, _, dep = self.renderer.render_view(orbit_pose(az, elevation), size)  # canonical frame
            cover = dep[..., 3] > 0
            uv = dep[..., 1:3]
            col = bilinear(tex, (uv[..., 0] * tw - 0.5).ravel(), ((1 - uv[..., 1]) * th - 0.5).ravel())
            out.append(np.where(cover[..., None], col.reshape(size, size, 3), 1.0))
        return np.clip(np.stack(out), 0, 1).astype(np.float32)


def to_pil(tex: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(tex * 255 + 0.5, 0, 255).astype(np.uint8))


def textured_mesh(prep, albedo: Image.Image, orm: Image.Image | None) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=albedo,
        metallicRoughnessTexture=orm,
        metallicFactor=1.0 if orm is not None else 0.0,
        roughnessFactor=1.0 if orm is not None else 0.8,
    )
    return trimesh.Trimesh(
        vertices=prep.original_vertices(),
        faces=prep.faces,
        vertex_normals=prep.normals,
        visual=trimesh.visual.TextureVisuals(uv=prep.uvs, material=material),
        process=False,
    )
