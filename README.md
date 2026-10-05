# ComfyUI-UltraTex

ComfyUI nodes for [UltraTex](https://yiboz2001.github.io/UltraTex/) — *Unleashing 2K Multi-View Diffusion
for 3D Texturing* (SIGGRAPH Asia 2026). Give it an untextured mesh and a reference image; it generates
six 2048² texture views with FLUX.2-Klein-base-9B (or FLUX.1-dev) + the UltraTex LoRA, and bakes them
into a UV texture on the mesh (GLB).

The diffusion model, VAE and text encoder are loaded with the **stock ComfyUI loaders**, so ComfyUI's
memory management, fp8/GGUF weights and offloading all apply.

## Nodes

| Node | What it does |
|---|---|
| **UltraTex Load LoRA** | Applies an UltraTex LoRA. The stock *Load LoRA* node can't read these checkpoints (PEFT `lora_A.default` keys for FLUX.2, UNO `processor.*_lora` keys for FLUX.1). |
| **UltraTex Foreground VAE Decoder** | Replaces the VAE decoder with UltraTex's Foreground-Aware decoder (`decoder.pt`). |
| **UltraTex Prep (mesh + reference)** | Normalises the mesh, UV-unwraps it (xatlas, or keeps existing UVs), renders the 6 canonical G-buffer TexVerse views (world normals + masks) and aligns the reference image to the front view. |
| **UltraTex Sampler** | UltraTex sampling: background-token dropping, normal-map + reference conditioning, UNO schedule. Outputs the 6 views (in view order front, left, back, right, top, bottom), their masks and the 2×3 atlas. |
| **UltraTex Bake Texture** | Back-projects the views into a UV texture (visibility + view-angle weighted blend, occlusion fill), saves `output/<prefix>_XXXXX_.glb` (+ albedo / ORM png) and returns a TRIMESH, the texture, an 8-view preview, the GLB path and `model_3d` (connect it straight to *Preview 3D*). |

## Models

| File | Folder | Source |
|---|---|---|
| `flux-2-klein-base-9b(-fp8).safetensors` | `models/diffusion_models` | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) — the **base** model, not the distilled Klein 9B |
| `qwen_3_8b*.safetensors` | `models/text_encoders` | ComfyUI Klein-9B text encoder (CLIP loader type `flux2`) |
| `flux2-vae.safetensors` | `models/vae` | FLUX.2 VAE |
| `flux2/lora/dit_lora.safetensors` (albedo), `flux2_mr/lora/dit_lora.safetensors` (metallic-roughness) | `models/loras` (rename, e.g. `ultratex_flux2.safetensors`, `ultratex_flux2_mr.safetensors`) | [ModelScope Yibo-Zhang/UltraTex](https://www.modelscope.ai/models/Yibo-Zhang/UltraTex) |
| `flux2/decoder.pt`, `flux1/decoder.pt` | `models/ultratex` (e.g. `ultratex_flux2_decoder.pt`) | same |

FLUX.1: `flux1-dev` in `models/diffusion_models`, `ae.safetensors` (FLUX.1 VAE), DualCLIPLoader with
`clip_l` + `t5xxl` (type `flux`), `flux1/lora/dit_lora.safetensors` and `flux1/decoder.pt`.

## Workflow (FLUX.2)

```
Load Diffusion Model (klein base 9B) ─ UltraTex Load LoRA ─────────┐
Load VAE (flux2-vae) ─ UltraTex Foreground VAE Decoder ────────────┤
Load CLIP (qwen 3 8B, flux2) ─ CLIP Text Encode ("")  ─ positive ──┤
Load Image (reference, RGBA) ─┬ image ─ UltraTex Prep ── prep ─────┼─ UltraTex Sampler ─ views ─ UltraTex Bake ─ model_3d ─ Preview 3D
                              └ mask ──┘   (mesh_path)        └────────────────────────────────────┘ prep
```

* The prompt should be **empty**: UltraTex is image-guided. Leave `negative` unconnected: with an empty
  prompt CFG is a no-op that doubles the cost of every step (the sampler also skips it automatically when
  the negative is identical to the positive).
* Reference image: the object on a transparent background (or connect a background-removal mask and set
  `mask_meaning` to `foreground`). A front view works best.

### Metallic-roughness

The same *UltraTex Sampler* predicts metallic-roughness when its model carries the **MR LoRA**
(`flux2_mr/lora/dit_lora.safetensors`) instead of the albedo one. Branch the diffusion model:

```
Load Diffusion Model ─┬ UltraTex Load LoRA (ultratex_flux2)    ─ UltraTex Sampler ─ views ─ Bake.albedo_views
                      └ UltraTex Load LoRA (ultratex_flux2_mr) ─ UltraTex Sampler ─ views ─ Bake.metallic_roughness_views
```

Both samplers share the same VAE (FLUX.2 foreground decoder), positive conditioning, prep and seed.
The bake writes a glTF ORM texture (G = roughness, B = metallic). MR maps are low-frequency, so the MR
sampler can run at 1024 even when albedo runs at 2048.

## Performance (RTX 3080 Laptop 16 GB, fp8 Klein base 9B, 25 steps)

| resolution | tokens in the DiT | time |
|---|---|---|
| 1024 | ~10k | ~3 min |
| 2048 | ~31k | ~25 min |

## Notes

* Camera / normal conventions were reverse-engineered from the G-buffer TexVerse samples: 6 views at
  distance 1.67, 35.5° FoV, mesh centred and scaled to a bounding radius of 0.5, world normals stored as
  Blender `(x, y, -z)`.
* UltraTex code is MIT licensed; the sparse-attention kernels it trains with are not used here yet
  (ComfyUI's attention backend is used).
