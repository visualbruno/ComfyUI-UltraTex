"""ComfyUI nodes for UltraTex (2K multi-view diffusion 3D texturing, SIGGRAPH Asia 2026).

Pipeline:  Load Diffusion Model + Load VAE + Load CLIP (stock ComfyUI)
           -> UltraTex Load LoRA, UltraTex Foreground VAE Decoder
           -> UltraTex Prep (mesh + reference) -> UltraTex Sampler -> UltraTex Bake
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import trimesh
from PIL import Image

import comfy.sd
import comfy.utils
import folder_paths

try:  # ComfyUI's 3D file type (Preview 3D input); absent in older ComfyUI versions
    from comfy_api.latest import Types

    File3D = Types.File3D
except (ImportError, AttributeError):
    File3D = None

from .ultratex_core import bake as bake_core
from .ultratex_core import lora as lora_core
from .ultratex_core import prep as prep_core
from .ultratex_core import sampling

log = logging.getLogger("UltraTex")

ULTRATEX_DIR = os.path.join(folder_paths.models_dir, "ultratex")
folder_paths.add_model_folder_path("ultratex", ULTRATEX_DIR)
if "ultratex" in folder_paths.folder_names_and_paths:
    paths, _ = folder_paths.folder_names_and_paths["ultratex"]
    folder_paths.folder_names_and_paths["ultratex"] = (paths, folder_paths.supported_pt_extensions)

CATEGORY = "UltraTex"


def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.cpu().numpy() * 255 + 0.5, 0, 255).astype(np.uint8)


def _uint8_to_image(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def _resolve_path(path: str) -> str:
    path = path.strip().strip('"')
    candidates = [path, os.path.join(folder_paths.get_input_directory(), path), os.path.join(folder_paths.get_output_directory(), path)]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"UltraTex: mesh file not found: {path!r} (absolute, or relative to the input/output folders)")


# =========================================================================== LoRA
class UltraTexLoraLoader:
    """Applies an UltraTex LoRA (PEFT FLUX.2 or UNO FLUX.1 format) as ComfyUI weight patches."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load an UltraTex LoRA (flux2/lora, flux2_mr/lora or flux1/lora dit_lora.safetensors). The stock LoRA loader cannot read these key formats."

    def load(self, model, lora_name, strength):
        family = sampling.model_family(model)
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        dm = model.model.diffusion_model
        shapes = {
            f"diffusion_model.single_blocks.{i}.linear1.weight": (blk.linear1.out_features,)
            for i, blk in enumerate(dm.single_blocks)
            if hasattr(blk.linear1, "out_features")
        }
        converted, fmt = lora_core.convert(sd, shapes)
        expected = {"peft": "flux2", "uno": "flux1"}.get(fmt)
        if expected is not None and expected != family:
            raise ValueError(f"UltraTex: this LoRA is for {expected.upper()} but the model is {family.upper()}")
        new_model, _ = comfy.sd.load_lora_for_models(model, None, converted, strength, 0.0)
        n_expected = len(converted) // 2 if fmt in ("peft", "uno") else None
        n_patched = len(new_model.patches) - len(model.patches)
        log.info("UltraTex LoRA %s (%s format): %d layers patched%s", lora_name, fmt, n_patched,
                 f" of {n_expected}" if n_expected else "")
        if n_expected and n_patched < n_expected:
            log.warning("UltraTex LoRA: %d LoRA layers did not match the model", n_expected - n_patched)
        return (new_model,)


# =========================================================================== VAE decoder
class UltraTexForegroundDecoder:
    """Swaps the VAE decoder for UltraTex's Foreground-Aware decoder (flux1/decoder.pt or flux2/decoder.pt)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_name": (folder_paths.get_filename_list("ultratex") + folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Put decoder.pt files in models/ultratex (or models/vae). FLUX.1 and FLUX.2 decoders are not interchangeable."

    def apply(self, vae, decoder_name):
        path = folder_paths.get_full_path("ultratex", decoder_name) or folder_paths.get_full_path_or_raise("vae", decoder_name)
        dec = comfy.utils.load_torch_file(path, safe_load=True)
        sd = vae.get_sd()
        sd = {k: v.detach().clone() for k, v in sd.items()}
        replaced = 0
        for k, v in dec.items():
            target = k if k.startswith("post_quant_conv.") else f"decoder.{k}"
            if target not in sd:
                raise ValueError(f"UltraTex: decoder key {k!r} has no counterpart in this VAE (wrong backbone?)")
            if tuple(sd[target].shape) != tuple(v.shape):
                raise ValueError(f"UltraTex: decoder {decoder_name} does not fit this VAE ({target}: {tuple(v.shape)} vs {tuple(sd[target].shape)})")
            sd[target] = v.to(sd[target].dtype)
            replaced += 1
        new_vae = comfy.sd.VAE(sd=sd)
        new_vae.throw_exception_if_invalid()
        log.info("UltraTex: replaced %d decoder tensors from %s", replaced, decoder_name)
        return (new_vae,)


# =========================================================================== Prep
class UltraTexPrep:
    """Normalises + UV-unwraps the mesh, renders the 6 TexVerse views and aligns the reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "mesh_path": ("STRING", {"default": "", "tooltip": "Mesh file (glb/obj/ply/...), absolute or relative to the input folder. Ignored when a TRIMESH is connected."}),
                "render_size": ("INT", {"default": 4096, "min": 512, "max": 4096, "step": 512, "tooltip": "TexVerse renders are 4096 px, then resized to the sampling resolution."}),
                "atlas_size": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512, "tooltip": "Texture resolution the xatlas charts are packed for."}),
                "uv_mode": (["comfy_gpu", "xatlas", "keep_existing"], {"default": "comfy_gpu", "tooltip": "comfy_gpu: ComfyUI's GPU unwrapper (same as the 'Unwrap Mesh UVs' node), seconds instead of minutes. xatlas: CPU, slow on dense meshes. keep_existing: the mesh's own UVs."}),
                "reference_alignment": (["fit_front_silhouette", "center"], {"tooltip": "fit_front_silhouette: crop the reference to its mask and fit it on the front view's silhouette."}),
                "mask_meaning": (["background (LoadImage)", "foreground"], {"tooltip": "LoadImage's MASK is 1 on transparent pixels; background-removal nodes usually output 1 on the object."}),
            },
            "optional": {
                "reference_mask": ("MASK",),
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("ULTRATEX_PREP", "IMAGE", "IMAGE")
    RETURN_NAMES = ("prep", "normal_views", "reference")
    FUNCTION = "prep"
    CATEGORY = CATEGORY

    def prep(self, reference_image, mesh_path, render_size, atlas_size, uv_mode, reference_alignment, mask_meaning, reference_mask=None, trimesh=None):
        mesh_obj = trimesh if trimesh is not None else _load_mesh(_resolve_path(mesh_path))
        rgb = _image_to_uint8(reference_image[0])
        if reference_mask is not None:
            m = reference_mask[0].cpu().numpy()
            if m.shape != rgb.shape[:2]:
                m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(rgb.shape[1::-1], Image.Resampling.BILINEAR)) / 255.0
            alpha = 1.0 - m if mask_meaning.startswith("background") else m
        elif rgb.shape[-1] == 4:
            alpha = rgb[..., 3] / 255.0
        else:
            raise ValueError("UltraTex Prep: connect reference_mask (LoadImage's MASK or a background-removal mask)")
        rgba = np.dstack([rgb[..., :3], np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
        prep = prep_core.run_prep(mesh_obj, rgba, render_size, atlas_size, uv_mode, reference_alignment)

        preview_size = min(render_size, 1024)
        normals = np.stack([np.asarray(Image.fromarray(v).resize((preview_size, preview_size), Image.Resampling.BILINEAR)) for v in prep.normal_rgba])
        normals = normals[..., :3] * (normals[..., 3:] > 0) + 255 * (normals[..., 3:] == 0)
        ref = Image.fromarray(prep.reference_rgba).resize((preview_size, preview_size), Image.Resampling.LANCZOS)
        ref_white = Image.new("RGB", ref.size, (255, 255, 255))
        ref_white.paste(ref, mask=ref.getchannel("A"))
        return (prep, _uint8_to_image(normals.astype(np.uint8)), _uint8_to_image(np.asarray(ref_white))[None])


def _load_mesh(path: str):
    return trimesh.load(path, process=False)


# =========================================================================== Sampler
class UltraTexSampler:
    """Generates the 6 texture views (albedo or metallic-roughness, depending on the LoRA)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "FLUX.2-Klein base 9B or FLUX.1-dev with an UltraTex LoRA applied."}),
                "vae": ("VAE", {"tooltip": "Matching VAE, ideally with the UltraTex foreground decoder applied."}),
                "positive": ("CONDITIONING", {"tooltip": "CLIP Text Encode with an EMPTY prompt (UltraTex is image-guided)."}),
                "prep": ("ULTRATEX_PREP",),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 16, "tooltip": "Per-view resolution. UltraTex is trained at 2048."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "FLUX.1: distilled guidance. FLUX.2: text CFG scale, only used when a negative is connected."}),
                "drop_background_tokens": ("BOOLEAN", {"default": True, "tooltip": "UltraTex Background Token Dropping (huge speed-up, as trained)."}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "memory_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 1.0, "tooltip": "VRAM kept free for activations, per token (x hidden x dtype). 0 = auto (12 with token chunking, 28 without). Raise it on out-of-memory errors."}),
                "attention": (["ultratex_sparse", "comfy_default"], {"default": "ultratex_sparse", "tooltip": "ultratex_sparse: UltraTex's block-sparse top-k attention (Triton), the attention UltraTex was trained with; ~4x faster than flash attention at 2048. comfy_default: dense attention from ComfyUI's backend."}),
                "sparse_topk": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "Fraction of key blocks each query block attends to (UltraTex default 0.2; 1.0 = dense)."}),
                "token_chunk": ("INT", {"default": 8192, "min": 0, "max": 131072, "step": 1024, "tooltip": "Compute the DiT MLPs in chunks of this many tokens to cut activation memory (0 = off). Same result, keeps the whole model on the GPU at 2048."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("views", "masks", "atlas")
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, vae, positive, prep, resolution, steps, seed, guidance, drop_background_tokens, negative=None,
               memory_factor=0.0, attention="ultratex_sparse", sparse_topk=0.2, token_chunk=8192):
        if resolution % 16:
            raise ValueError("UltraTex: resolution must be a multiple of 16")
        views, masks = sampling.sample(model, vae, positive, negative, prep, resolution, steps, seed, guidance,
                                       drop_background_tokens, memory_factor, attention, sparse_topk, token_chunk)
        rows = [torch.cat([views[v] for v in row], dim=1) for row in sampling.GRID_ORDER]
        atlas = torch.cat(rows, dim=0)[None]
        return (views, masks, atlas)


# =========================================================================== Bake
class UltraTexBake:
    """Back-projects the 6 views into a UV texture and saves a textured GLB in the output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": ("ULTRATEX_PREP",),
                "albedo_views": ("IMAGE",),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512}),
                "filename_prefix": ("STRING", {"default": "UltraTex/mesh"}),
                "view_weight_power": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 16.0, "step": 0.5, "tooltip": "Blend weight = cos(normal, view)^power."}),
                "best_view_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = pure weighted blend, 1 = best view only (sharper, more visible seams)."}),
                "edge_feather_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 1.0, "tooltip": "Down-weight pixels near silhouettes / occlusion edges."}),
            },
            "optional": {
                "metallic_roughness_views": ("IMAGE", {"tooltip": "Views from the MR LoRA: G = roughness, B = metallic."}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "STRING", "FILE_3D_GLB")
    RETURN_NAMES = ("trimesh", "albedo_texture", "preview", "glb_path", "model_3d")
    OUTPUT_TOOLTIPS = (
        "Textured trimesh (original scale and position).",
        "Baked albedo texture.",
        "Unlit renders from 8 azimuths.",
        "GLB path relative to the output folder.",
        "The saved GLB as a 3D file: connect to Preview 3D.",
    )
    FUNCTION = "bake"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def bake(self, prep, albedo_views, texture_size, filename_prefix, view_weight_power, best_view_mix, edge_feather_px, metallic_roughness_views=None):
        settings = bake_core.BakeSettings(power=view_weight_power, best_view_mix=best_view_mix, edge_px=edge_feather_px)
        baker = bake_core.Baker(prep, texture_size)
        try:
            albedo = bake_core.to_pil(baker.bake(*self._views(albedo_views, prep), settings))
            orm = None
            if metallic_roughness_views is not None:
                orm_np = np.asarray(bake_core.to_pil(baker.bake(*self._views(metallic_roughness_views, prep), settings))).copy()
                orm_np[..., 0] = 255  # R = occlusion (unused)
                orm = Image.fromarray(orm_np)
            preview = baker.preview(np.asarray(albedo).astype(np.float32) / 255.0)
        finally:
            baker.release()

        mesh = bake_core.textured_mesh(prep, albedo, orm)
        out_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        name = f"{filename}_{counter:05}_"
        glb_file = os.path.join(full_folder, name + ".glb")
        mesh.export(glb_file)
        albedo.save(os.path.join(full_folder, name + "_albedo.png"))
        if orm is not None:
            orm.save(os.path.join(full_folder, name + "_orm.png"))
        rel = os.path.join(subfolder, name + ".glb").replace("\\", "/")
        log.info("UltraTex: saved %s", rel)
        albedo_t = _uint8_to_image(np.asarray(albedo))[None]
        model_3d = File3D(glb_file, file_format="glb") if File3D is not None else None
        return {"ui": {"text": [rel]}, "result": (mesh, albedo_t, torch.from_numpy(preview), rel, model_3d)}

    @staticmethod
    def _views(images: torch.Tensor, prep):
        if images.shape[0] != 6:
            raise ValueError(f"UltraTex Bake: expected 6 views, got {images.shape[0]}")
        views = images[..., :3].cpu().numpy().astype(np.float32)
        size = views.shape[1]
        masks = np.stack([
            np.asarray(Image.fromarray(prep.normal_rgba[v, ..., 3]).resize((size, size), Image.Resampling.BILINEAR)) > 0
            for v in range(6)
        ])
        return views, masks

class UltraTexLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}),
                "only_vertices_and_faces": ("BOOLEAN",{"default":False}),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path, only_vertices_and_faces = False):
        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        mesh = trimesh.load(glb_path, force="mesh")
        
        if only_vertices_and_faces:
            mesh = trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces)
        
        return (mesh,)

NODE_CLASS_MAPPINGS = {
    "UltraTexLoraLoader": UltraTexLoraLoader,
    "UltraTexForegroundDecoder": UltraTexForegroundDecoder,
    "UltraTexPrep": UltraTexPrep,
    "UltraTexSampler": UltraTexSampler,
    "UltraTexBake": UltraTexBake,
    "UltraTexLoadMesh": UltraTexLoadMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraTexLoraLoader": "UltraTex Load LoRA",
    "UltraTexForegroundDecoder": "UltraTex Foreground VAE Decoder",
    "UltraTexPrep": "UltraTex Prep (mesh + reference)",
    "UltraTexSampler": "UltraTex Sampler",
    "UltraTexBake": "UltraTex Bake Texture",
    "UltraTexLoadMesh": "UltraText Load Mesh",
}
