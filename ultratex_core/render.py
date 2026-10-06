"""Mesh normalisation and offscreen G-buffer rendering in G-buffer TexVerse conventions.

Conventions (reverse-engineered from the G-buffer TexVerse samples shipped with UltraTex):
  * Mesh is centred on its bbox centre and scaled so the farthest vertex is 0.5 from it.
  * Camera poses are OpenGL camera-to-world matrices in Blender's Z-up world
    (camera looks down -Z, +Y up). Blender world = (x, -z, y) of glTF Y-up coordinates.
  * 6 canonical views at distance 1.67, intrinsics fx = fy = 6400 at 4096 px (fov ~35.5 deg).
  * bump_normal_world rgb = (nx, ny, -nz)_blender * 0.5 + 0.5
  * 2x3 atlas layout: row 1 = views 0, 1, 3 ; row 2 = views 2, 4, 5
"""

from __future__ import annotations

import numpy as np
import trimesh

CAM_DIST = 1.67
FOCAL_OVER_SIZE = 6400.0 / 4096.0  # fx / image width
FOV_Y = 2 * np.arctan(0.5 / FOCAL_OVER_SIZE)

# 3x4 camera-to-world, Blender frame (pose/00{0..5}.npy of every TexVerse asset)
POSES = np.array(
    [
        [[1, 0, 0, 0], [0, 0, -1, -CAM_DIST], [0, 1, 0, 0]],
        [[0, 0, -1, -CAM_DIST], [-1, 0, 0, 0], [0, 1, 0, 0]],
        [[-1, 0, 0, 0], [0, 0, 1, CAM_DIST], [0, 1, 0, 0]],
        [[0, 0, 1, CAM_DIST], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, CAM_DIST]],
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -CAM_DIST]],
    ],
    dtype=np.float64,
)
VIEW_NAMES = ["front", "left", "back", "right", "top", "bottom"]
GRID_ORDER = [[0, 1, 3], [2, 4, 5]]
GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def to_single_mesh(obj) -> trimesh.Trimesh:
    """Flatten a trimesh Scene / Trimesh into one Trimesh (keeps UVs when present)."""
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry() if hasattr(obj, "to_geometry") else obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh.Trimesh or Scene, got {type(obj).__name__}")
    return obj


def normalize(vertices: np.ndarray, faces: np.ndarray | None = None):
    """Return (normalised vertices, centre, scale) with normalised = (v - centre) * scale.

    Only vertices referenced by `faces` define the bounds (stray vertices are ignored).
    """
    used = vertices if faces is None else vertices[np.unique(faces)]
    lo, hi = used.min(0), used.max(0)
    centre = (lo + hi) / 2
    radius = np.linalg.norm(used - centre, axis=1).max()
    scale = 0.5 / radius
    return (vertices - centre) * scale, centre, scale


def perspective(fov_y: float, near: float = 0.05, far: float = 10.0) -> np.ndarray:
    f = 1.0 / np.tan(fov_y / 2)
    return np.array(
        [
            [f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def orbit_pose(azimuth_deg: float, elevation_deg: float, dist: float = CAM_DIST) -> np.ndarray:
    """3x4 Blender-frame camera-to-world looking at the origin. azimuth 0 = front (-Y)."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    eye = dist * np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    back = eye / np.linalg.norm(eye)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.stack([right, up, back, eye], axis=1)


def rig_rotation(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    """Object rotation (glTF frame) that makes the canonical cameras see the object from a rig turned by
    `azimuth_deg` around the vertical axis and tilted by `elevation_deg` (cameras higher for > 0)."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])  # about +Y (up)
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])  # about +X
    return rx @ ry


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


# direction from the object to canonical cameras 0..3 (front, left, back, right) in the glTF frame
_SIDE_DIRS = np.array([[0, 0, 1], [-1, 0, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def per_view_rigs(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """(6, 3, 3) object rotations: each side camera sees the object turned by `azimuth_deg` AND from
    `elevation_deg` above (every side view raised, no roll); top / bottom only get the azimuth.
    Unlike a rigid rig tilt, the 6 views no longer share one object orientation (non-canonical layout)."""
    base = rig_rotation(azimuth_deg, 0.0)
    rigs = np.repeat(base[None], 6, axis=0)
    up = np.array([0.0, 1.0, 0.0])
    for v, d in enumerate(_SIDE_DIRS):
        # tipping the object's top towards camera v by `elevation` == camera v raised by `elevation`
        rigs[v] = _axis_angle(np.cross(up, d), np.radians(elevation_deg)) @ base
    return rigs


def rig_for_view(rig, view: int):
    """A rig is None, one (3, 3) rotation for all views, or a (6, 3, 3) per-view stack."""
    if rig is None:
        return None
    rig = np.asarray(rig)
    return rig[view] if rig.ndim == 3 else rig


def view_matrix(view, rig: np.ndarray | None = None) -> np.ndarray:
    """World (glTF frame) -> camera, for a canonical view index or a 3x4 camera-to-world pose.

    `rig` is an object rotation applied before the canonical cameras (rotated rig)."""
    c2w = np.eye(4)
    c2w[:3, :4] = POSES[view] if np.isscalar(view) else view
    to_bl = np.eye(4)
    to_bl[:3, :3] = GLTF_TO_BLENDER
    mv = np.linalg.inv(c2w) @ to_bl
    if rig is not None:
        r4 = np.eye(4)
        r4[:3, :3] = rig
        mv = mv @ r4
    return mv


def encode_normal_world(n_gltf: np.ndarray) -> np.ndarray:
    n_bl = n_gltf @ GLTF_TO_BLENDER.T
    n_bl[..., 2] *= -1
    return n_bl * 0.5 + 0.5


def project(points_gltf: np.ndarray, view, size: int, rig: np.ndarray | None = None):
    """Project glTF-frame points into view pixels. Returns (u, v, depth)."""
    mv = view_matrix(view, rig)
    p = points_gltf @ mv[:3, :3].T + mv[:3, 3]
    depth = -p[..., 2]
    f = FOCAL_OVER_SIZE * size
    return size / 2 + f * p[..., 0] / depth, size / 2 - f * p[..., 1] / depth, depth


_CTX = None


def _context():
    global _CTX
    if _CTX is None:
        import moderngl

        _CTX = moderngl.create_standalone_context(require=330)
    return _CTX


class GLRenderer:
    """moderngl G-buffer renderer (float32 MRT, no AA => binary alpha like TexVerse)."""

    VS = """
    #version 330
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform int uv_mode;
    in vec3 in_pos;
    in vec3 in_nrm;
    in vec2 in_uv;
    out vec3 v_pos;
    out vec3 v_nrm;
    out vec2 v_uv;
    out float v_depth;
    void main() {
        v_pos = in_pos;
        v_nrm = in_nrm;
        v_uv = in_uv;
        v_depth = -(mv * vec4(in_pos, 1.0)).z;
        gl_Position = uv_mode == 1 ? vec4(in_uv * 2.0 - 1.0, 0.0, 1.0) : mvp * vec4(in_pos, 1.0);
    }
    """
    FS = """
    #version 330
    in vec3 v_pos;
    in vec3 v_nrm;
    in vec2 v_uv;
    in float v_depth;
    layout(location = 0) out vec4 o_pos;
    layout(location = 1) out vec4 o_nrm;
    layout(location = 2) out vec4 o_depth;
    void main() {
        float len_n = length(v_nrm);  // can vanish between opposite vertex normals
        o_pos = vec4(v_pos, 1.0);
        o_nrm = vec4(len_n > 1e-12 ? v_nrm / len_n : vec3(0.0), 1.0);
        o_depth = vec4(v_depth, v_uv, 1.0);
    }
    """

    def __init__(self, vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None):
        self.ctx = _context()
        self.prog = self.ctx.program(vertex_shader=self.VS, fragment_shader=self.FS)
        if uvs is None:
            uvs = np.zeros((len(vertices), 2))
        data = np.concatenate([vertices, normals, uvs], axis=1).astype("f4")
        self.vbo = self.ctx.buffer(data.tobytes())
        self.ibo = self.ctx.buffer(np.ascontiguousarray(faces, dtype="i4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")], self.ibo)

    def release(self):
        for obj in (self.vao, self.vbo, self.ibo, self.prog):
            obj.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def _render(self, size: int, mvp: np.ndarray, mv: np.ndarray, uv_mode: int):
        import moderngl

        ctx = self.ctx
        atts = [ctx.texture((size, size), 4, dtype="f4") for _ in range(3)]
        depth = ctx.depth_renderbuffer((size, size))
        fbo = ctx.framebuffer(color_attachments=atts, depth_attachment=depth)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.disable(moderngl.CULL_FACE)
        if uv_mode:
            ctx.disable(moderngl.DEPTH_TEST)
        else:
            ctx.enable(moderngl.DEPTH_TEST)
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["mv"].write(mv.T.astype("f4").tobytes())
        self.prog["uv_mode"].value = uv_mode
        self.vao.render(moderngl.TRIANGLES)
        out = []
        for att in atts:
            out.append(np.frombuffer(att.read(), dtype="f4").reshape(size, size, 4)[::-1].copy())
            att.release()
        depth.release()
        fbo.release()
        return out  # pos, nrm, (depth, u, v); alpha channel = coverage

    def render_view(self, view, size: int, rig: np.ndarray | None = None):
        """`view` is a canonical view index or a 3x4 Blender-frame camera-to-world pose; `rig` rotates the
        object first. Position / normal outputs stay in the unrotated object frame."""
        mv = view_matrix(view, rig)
        return self._render(size, perspective(FOV_Y) @ mv, mv, 0)

    def render_uv(self, size: int):
        """Rasterise in UV space -> per-texel glTF position and normal (row 0 = v = 1)."""
        return self._render(size, np.eye(4), np.eye(4), 1)
