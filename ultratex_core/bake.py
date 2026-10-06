"""Back-project the 6 generated views into a UV texture and build a textured trimesh.

For every texel (rasterised in UV space) and every canonical view:
  visible = texel depth matches the view depth buffer
  weight  = cos(normal, view dir)^power * ramp(distance to silhouette / occlusion edges)
Texels seen by no view take the colour of the nearest seen texels in 3D; chart padding is
filled by UV-space nearest-neighbour dilation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .render import GLRenderer, orbit_pose, project, rig_for_view, view_matrix

log = logging.getLogger("UltraTex")


@dataclass
class BakeSettings:
    power: float = 4.0
    min_cos: float = 0.05
    depth_eps: float = 0.004
    edge_jump: float = 0.01
    edge_px: float = 6.0
    best_view_mix: float = 0.5
    min_weight: float = 1e-4
    white_rim_px: float = 12.0  # at 1024 px views; near-white pixels this close inside a silhouette are dropped


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


class Baker:
    def __init__(self, prep, tex_size: int):
        self.prep = prep
        self.tex_size = tex_size
        self.renderer = GLRenderer(prep.vertices, prep.normals, prep.faces, prep.uvs)
        tpos, tnrm, tdep = self.renderer.render_uv(tex_size)
        self.cover = tdep[..., 3] > 0
        self.texel_pos = tpos[..., :3][self.cover].astype(np.float64)
        n = np.nan_to_num(tnrm[..., :3][self.cover].astype(np.float64))  # zero normal => weight 0
        self.texel_nrm = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
        self._view_cache = {}
        log.info("UltraTex bake: UV coverage %.3f of %d^2 texels", self.cover.mean(), tex_size)

    def release(self):
        self.renderer.release()

    def _view_geometry(self, view: int, size: int, mask: np.ndarray, s: BakeSettings, rig: np.ndarray | None):
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
        visible = inside & cover[vi, ui] & (np.abs(d - depth[vi, ui]) < s.depth_eps)
        cam = np.linalg.inv(view_matrix(view, rig))[:3, 3]  # camera centre in the object frame
        to_cam = cam[None] - self.texel_pos
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        cos = np.einsum("ij,ij->i", self.texel_nrm, to_cam)
        ramp = np.clip(edge_dist[vi, ui] / s.edge_px, 0.0, 1.0)
        ok = visible & (cos > s.min_cos)
        w = np.where(ok, np.clip(cos, 0, 1) ** s.power * ramp, 0.0)
        quality = np.where(ok, np.clip(cos, 0, 1) * ramp, 0.0)  # how head-on (and away from edges)
        return u, v, w, quality

    def bake(self, view_sets, s: BakeSettings):
        """view_sets: [(views (6, H, W, 3) float [0,1], masks (6, H, W) bool, rig or None[, view indices]), ...]

        Returns (texture (T, T, 3) float, confidence (T, T) float = best per-texel view quality)."""
        n = len(self.texel_pos)
        acc = np.zeros((n, 3))
        wsum = np.zeros(n)
        best_w = np.zeros(n)
        best_c = np.zeros((n, 3))
        best_q = np.zeros(n)
        for views, masks, rig, *rest in view_sets:
            use = rest[0] if rest else None  # optional subset of the 6 views
            for view in range(6):
                if use is not None and view not in use:
                    continue
                size = views[view].shape[0]
                valid = _drop_white_rim(views[view], masks[view], s.white_rim_px)
                u, v, w, q = self._view_geometry(view, size, valid, s, rig)
                col = bilinear(_extend_into_background(views[view], valid), u - 0.5, v - 0.5)
                acc += w[:, None] * col
                wsum += w
                better = w > best_w
                best_w[better], best_c[better] = w[better], col[better]
                np.maximum(best_q, q, out=best_q)
        seen = wsum > s.min_weight
        colors = np.zeros((n, 3))
        colors[seen] = acc[seen] / wsum[seen, None]
        colors[seen] = (1 - s.best_view_mix) * colors[seen] + s.best_view_mix * best_c[seen]
        log.info("UltraTex bake: %.3f of texels seen; filling the rest from nearest seen texels", seen.mean())
        if (~seen).any() and seen.any():
            _, idx = cKDTree(self.texel_pos[seen]).query(self.texel_pos[~seen], k=4)
            colors[~seen] = colors[seen][idx].mean(axis=1)
        tex = np.zeros((self.tex_size, self.tex_size, 3), np.float32)
        tex[self.cover] = colors
        conf = np.zeros((self.tex_size, self.tex_size), np.float32)
        conf[self.cover] = best_q
        _, (iy, ix) = ndimage.distance_transform_edt(~self.cover, return_indices=True)
        return tex[iy, ix], conf[iy, ix]

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
