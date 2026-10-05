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
| **UltraTex Prep (mesh + reference)** | Normalises the mesh, UV-unwraps it (`comfy_gpu`: ComfyUI's GPU unwrapper from the *Unwrap Mesh UVs* node, ~6 s for 500k faces; `xatlas`: CPU, ~4.6 min for 500k faces; or `keep_existing` UVs), renders the 6 canonical G-buffer TexVerse views (world normals + masks) and aligns the reference image to the front view. |
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

### Second pass: 4 more views at 45° / 135° / 225° / 315°

UltraTex only knows its 6-view layout, so extra views come from a second pass with the camera rig turned
45° (the object is rotated; the model still sees its canonical cameras). The second pass starts from the
first pass re-projected onto the rotated rig (SDEdit, `denoise` ~0.6) and keeps what the first pass saw
head-on (`keep_mask`), so both passes agree; the final bake blends all 12 views. On the test meshes this
raises the surface seen head-on (view angle < 45°) from 44% to 58% (dwarf) and 60% to 75% (anime girl).

```
UltraTex Prep ─ prep ──┬──────────────────────────────── Sampler 1 ─ views ─┬──────────────── Bake.albedo_views
                       │                                                    │    (Bake.prep = first prep)
                       ├─ UltraTex Rotate Rig (45) ─ prep ─┬─ Render Views.prep
                       │                                   │  Render Views.source_prep  ← first prep
                       │                                   │  Render Views.source_views ← Sampler 1 views
                       │                                   │       init_views, keep_mask ─┐
                       │                                   └─ Sampler 2.prep              │
                       │                                      Sampler 2 (denoise 0.6) ←───┘ ─ views ─ Bake.albedo_views_2
                       │                                                                         Bake.prep_2 ← rotated prep
```

Do **not** connect Sampler 1's `views` / `masks` straight into Sampler 2: they are seen from the other
rig (the sampler refuses misaligned `init_views`). Sampler 2 runs `denoise x steps` steps (0.6 x 25 ≈ 11).

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

## Performance (RTX 3080 Laptop 16 GB, fp8 Klein base 9B, 25 steps, ComfyUI dynamic VRAM)

Anime-girl test asset (T-pose; bulkier objects have more foreground tokens and are slower):

| resolution | tokens in the DiT | sampling | whole workflow (prep cached, 4K bake) |
|---|---|---|---|
| 1024 | ~10k | 118 s (4.7 s/step) | ~2.3 min |
| 2048 | ~32k | 472 s (18.9 s/step) | ~9.6 min |

Sampler optimisations (on by default):

* `attention = ultratex_sparse` — UltraTex's block-sparse top-k attention (vendored SLA Triton kernel,
  the attention UltraTex is trained with). 4.4x faster than flash-attention at 64k tokens. At ~32k
  tokens the linear layers dominate (~37 TFLOPs bf16, compute bound), so the step gain is ~20%; for
  bulky meshes at 2048 (60k+ tokens) attention dominates and the gain is much larger.
* `token_chunk = 8192` — DiT MLPs computed in token chunks (identical result up to bf16 rounding), so
  activation memory stays small and the whole model stays on the GPU at 2048.
* CFG is skipped when the negative equals the positive (empty prompts).
* `memory_factor = 0` (auto) sizes the VRAM reserve for the chunked activations.

Other levers: fewer `steps` (time is linear in steps), `sparse_topk` 0.1 (faster attention, may cost
detail), run the metallic-roughness sampler at 1024.

Where the time goes (2048, bulky 500k-face character, 62k DiT tokens, `ULTRATEX_PROFILE=1`):
single-block linear1 ~16 s, linear2 ~7 s, double blocks ~8 s, attention ~12.5 s per step, i.e. ~40-45 s
per step. The matmuls run at ~75% of the GPU's burst bf16 rate; on laptops sustained runs are limited
by thermal throttling (observed: 87 °C, SM clock 1110 of 2100 MHz), so cooling / power mode matters more
than any remaining software setting. Set the environment variable `ULTRATEX_PROFILE=1` before starting
ComfyUI to log this breakdown for every step (it adds CUDA syncs, so leave it off normally).

## Notes

* Camera / normal conventions were reverse-engineered from the G-buffer TexVerse samples: 6 views at
  distance 1.67, 35.5° FoV, mesh centred and scaled to a bounding radius of 0.5, world normals stored as
  Blender `(x, y, -z)`.
* UltraTex code is MIT licensed. `ultratex_core/sparse_attention/` is UltraTex's vendored copy of
  [SLA](https://github.com/thu-ml/SLA) (Apache-2.0, see its `LICENSE.txt` and `PATCHES.md`); it needs
  Triton (`triton-windows` on Windows). Without it the sampler falls back to ComfyUI's attention.
