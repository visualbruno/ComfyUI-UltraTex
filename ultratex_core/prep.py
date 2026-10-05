"""Turn an untextured mesh + reference image into UltraTex conditions (TexVerse 6-view layout)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image

from .render import GLRenderer, encode_normal_world, normalize, rig_rotation, to_single_mesh

log = logging.getLogger("UltraTex")


@dataclass
class UltraTexPrep:
    # normalised, UV-split mesh used for rendering and baking (glTF Y-up frame)
    vertices: np.ndarray  # (V, 3) float32
    normals: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32
    uvs: np.ndarray  # (V, 2) float32, OpenGL convention (v up)
    centre: np.ndarray  # normalised = (original - centre) * scale
    scale: float
    # 6 canonical views at render_size
    normal_rgba: np.ndarray  # (6, S, S, 4) uint8, bump_normal_world encoding, alpha = coverage
    reference_rgba: np.ndarray  # (S, S, 4) uint8, reference aligned to the front view
    render_size: int
    # object rotation (glTF frame) applied before the canonical cameras; identity = canonical rig
    rig: np.ndarray = field(default_factory=lambda: np.eye(3))
    rig_label: str = "canonical"

    @property
    def masks(self) -> np.ndarray:
        return self.normal_rgba[..., 3] > 0

    def original_vertices(self) -> np.ndarray:
        return self.vertices / self.scale + self.centre


def unwrap_xatlas(vertices: np.ndarray, faces: np.ndarray, resolution: int, padding: int = 4):
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = resolution
    pack.padding = padding
    pack.bilinear = True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmapping, new_faces, uvs = atlas[0]
    return vmapping, new_faces, uvs


def unwrap_comfy(vertices: np.ndarray, faces: np.ndarray, resolution: int, padding: int = 4):
    """ComfyUI's GPU UV unwrapper ("Unwrap Mesh UVs" node, pec segmenter): much faster than xatlas.

    Returns (vmapping, faces, uvs) like xatlas; UVs in trimesh/OpenGL convention (v up).
    Raises ImportError on ComfyUI versions without comfy_extras.mesh3d.
    """
    import torch

    import comfy.model_management as mm
    from comfy_extras import nodes_mesh_postprocess as mp

    device = mm.get_torch_device()
    mp._prepare_gpu_mesh_processing(device, len(faces) * 14 * 1024)
    vmapping, new_faces, uvs = mp._uv_unwrap(
        torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)).to(device),
        torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)).to(device),
        "pec", int(resolution), int(padding), 0.0,
    )
    uvs = np.array(uvs, dtype=np.float32, copy=True)
    uvs[:, 1] = 1.0 - uvs[:, 1]  # ComfyUI's unwrapper is v-down; trimesh / our GL path are v-up
    return np.asarray(vmapping), np.asarray(new_faces), uvs


def sanitize_normals(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Replace NaN / zero vertex normals (degenerate faces, unreferenced vertices, bad file normals).

    Bad normals are recomputed from the area-weighted normals of their valid faces; vertices that
    still have none (unreferenced or only touching degenerate faces) take their nearest valid neighbour's.
    """
    n = np.where(np.isfinite(normals), normals, 0.0)
    length = np.linalg.norm(n, axis=1)
    bad = length < 1e-12
    if bad.any():
        tri = vertices[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
        fn[~np.isfinite(fn).all(1)] = 0.0
        acc = np.zeros_like(vertices)
        for k in range(3):
            np.add.at(acc, faces[:, k], fn)
        n[bad] = acc[bad]
        length = np.linalg.norm(n, axis=1)
        still = length < 1e-12
        if still.any() and (~still).any():
            from scipy.spatial import cKDTree

            _, idx = cKDTree(vertices[~still]).query(vertices[still])
            n[still] = n[~still][idx]
        elif still.any():
            n[still] = (0.0, 0.0, 1.0)
        log.info("UltraTex prep: repaired %d invalid vertex normals (%d had no valid face)", bad.sum(), still.sum())
        length = np.linalg.norm(n, axis=1)
    return n / np.maximum(length, 1e-12)[:, None]


def prepare_mesh(mesh_obj, uv_mode: str, atlas_size: int):
    mesh = to_single_mesh(mesh_obj)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices, centre, scale = normalize(np.asarray(mesh.vertices, dtype=np.float64), faces)
    # file normals when present, smooth area-weighted normals otherwise (unchanged by normalisation)
    normals = sanitize_normals(vertices, faces, np.asarray(mesh.vertex_normals, dtype=np.float64))
    existing_uv = getattr(mesh.visual, "uv", None)
    if uv_mode == "keep_existing" and existing_uv is not None and len(existing_uv) == len(vertices):
        uvs = np.asarray(existing_uv, dtype=np.float32)
        log.info("UltraTex prep: keeping the mesh's existing UVs")
        return vertices, normals, faces, uvs, centre, scale
    if uv_mode == "keep_existing":
        log.warning("UltraTex prep: mesh has no usable UVs, falling back to the GPU unwrapper")
        uv_mode = "comfy_gpu"
    t = time.time()
    if uv_mode == "comfy_gpu":
        try:
            vmapping, faces_uv, uvs = unwrap_comfy(vertices, faces, atlas_size)
        except ImportError as exc:
            log.warning("UltraTex prep: ComfyUI's UV unwrapper is unavailable (%s), using xatlas", exc)
            uv_mode = "xatlas"
    if uv_mode == "xatlas":
        vmapping, faces_uv, uvs = unwrap_xatlas(vertices, faces, atlas_size)
    log.info("UltraTex prep: %s unwrap of %d faces took %.1fs", uv_mode, len(faces), time.time() - t)
    return vertices[vmapping], normals[vmapping], faces_uv.astype(np.int64), uvs.astype(np.float32), centre, scale


def render_conditions(vertices, normals, faces, size: int, rig: np.ndarray | None = None) -> np.ndarray:
    """6 canonical views of the (rig-rotated) object; normals are expressed in the rotated world frame,
    which is what the model sees for an object standing in that orientation."""
    out = np.zeros((6, size, size, 4), np.uint8)
    with GLRenderer(vertices, normals, faces) as renderer:
        for view in range(6):
            _, nrm, _ = renderer.render_view(view, size, rig)
            alpha = nrm[..., 3] > 0
            n = np.nan_to_num(nrm[..., :3])
            if rig is not None:
                n = n @ rig.T
            n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
            rgb = np.clip(encode_normal_world(n) * 255 + 0.5, 0, 255).astype(np.uint8)
            out[view, ..., :3] = np.where(alpha[..., None], rgb, 0)
            out[view, ..., 3] = alpha.astype(np.uint8) * 255
    return out


def rotate_rig(prep: UltraTexPrep, azimuth: float, elevation: float) -> UltraTexPrep:
    """Same mesh / UVs / reference, seen by the canonical cameras with the object rotated (second pass)."""
    rig = rig_rotation(azimuth, elevation)
    normal_rgba = render_conditions(prep.vertices, prep.normals, prep.faces, prep.render_size, rig)
    return replace(prep, rig=rig, normal_rgba=normal_rgba, rig_label=f"az {azimuth:g} el {elevation:g}")


def align_reference(rgba: np.ndarray, front_alpha: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Crop the reference to its alpha bbox and fit it into the front silhouette bbox (or the frame)."""
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        raise ValueError("UltraTex prep: the reference mask is empty")
    ref = Image.fromarray(rgba).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if mode == "fit_front_silhouette":
        fy, fx = np.nonzero(front_alpha)
        k = size / front_alpha.shape[0]
        bx0, bx1, by0, by1 = fx.min() * k, (fx.max() + 1) * k, fy.min() * k, (fy.max() + 1) * k
    else:  # centre in frame with the TexVerse margin (~92% of the frame)
        bx0, by0 = 0.04 * size, 0.04 * size
        bx1, by1 = 0.96 * size, 0.96 * size
    bw, bh = bx1 - bx0, by1 - by0
    s = min(bw / ref.width, bh / ref.height)
    w, h = max(1, round(ref.width * s)), max(1, round(ref.height * s))
    ref = ref.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(ref, (round(bx0 + (bw - w) / 2), round(by0 + (bh - h) / 2)))
    return np.asarray(canvas)


def run_prep(mesh_obj, reference_rgba: np.ndarray, render_size: int, atlas_size: int, uv_mode: str, ref_align: str) -> UltraTexPrep:
    vertices, normals, faces, uvs, centre, scale = prepare_mesh(mesh_obj, uv_mode, atlas_size)
    log.info("UltraTex prep: %d verts / %d faces, normalised extents %s", len(vertices), len(faces), np.ptp(vertices, 0).round(3))
    normal_rgba = render_conditions(vertices, normals, faces, render_size)
    ref = align_reference(reference_rgba, normal_rgba[0, ..., 3] > 0, render_size, ref_align)
    return UltraTexPrep(
        vertices=vertices.astype(np.float32),
        normals=normals.astype(np.float32),
        faces=faces.astype(np.int32),
        uvs=uvs,
        centre=centre,
        scale=float(scale),
        normal_rgba=normal_rgba,
        reference_rgba=ref,
        render_size=render_size,
    )
