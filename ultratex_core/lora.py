"""Convert UltraTex LoRA checkpoints into ComfyUI's generic `<key>.lora_up/lora_down.weight` format.

* FLUX.2 (PEFT):  base_model.model.<module>.lora_{A,B}.default.weight, lora_alpha == rank (scale 1)
* FLUX.1 (UNO):   <block>.processor.<name>.{down,up}.weight, network_alpha None (scale 1)
      double: qkv_lora1 -> img_attn.qkv, proj_lora1 -> img_attn.proj,
              qkv_lora2 -> txt_attn.qkv, proj_lora2 -> txt_attn.proj
      single: qkv_lora  -> linear1 (qkv rows only; mlp rows get a zero up-projection)
              proj_lora -> linear2
"""

from __future__ import annotations

import torch

PEFT_PREFIX = "base_model.model."
UNO_DOUBLE = {
    "qkv_lora1": "img_attn.qkv",
    "proj_lora1": "img_attn.proj",
    "qkv_lora2": "txt_attn.qkv",
    "proj_lora2": "txt_attn.proj",
}
UNO_SINGLE = {"qkv_lora": "linear1", "proj_lora": "linear2"}


def detect_format(sd: dict) -> str:
    keys = list(sd)
    if any(k.startswith(PEFT_PREFIX) and ".lora_A." in k for k in keys):
        return "peft"
    if any(".processor." in k and (k.endswith(".down.weight") or k.endswith(".up.weight")) for k in keys):
        return "uno"
    return "unknown"


def _peft(sd: dict) -> dict:
    out = {}
    for key, down in sd.items():
        if ".lora_A." not in key:
            continue
        module = key[len(PEFT_PREFIX):].split(".lora_A.")[0]
        up = sd[key.replace(".lora_A.", ".lora_B.")]
        out[f"diffusion_model.{module}.lora_down.weight"] = down
        out[f"diffusion_model.{module}.lora_up.weight"] = up
    return out


def _uno(sd: dict, model_sd_shapes: dict) -> dict:
    out = {}
    for key, down in sd.items():
        if not key.endswith(".down.weight"):
            continue
        block, name = key[: -len(".down.weight")].split(".processor.")
        up = sd[key.replace(".down.weight", ".up.weight")]
        table = UNO_DOUBLE if block.startswith("double_blocks") else UNO_SINGLE
        target = f"diffusion_model.{block}.{table[name]}"
        full_rows = model_sd_shapes.get(f"{target}.weight", (up.shape[0],))[0]
        if full_rows != up.shape[0]:  # single-block qkv LoRA covers only the first 3H rows of linear1
            padded = torch.zeros(full_rows, up.shape[1], dtype=up.dtype)
            padded[: up.shape[0]] = up
            up = padded
        out[f"{target}.lora_down.weight"] = down
        out[f"{target}.lora_up.weight"] = up
    return out


def convert(sd: dict, model_sd_shapes: dict) -> tuple[dict, str]:
    fmt = detect_format(sd)
    if fmt == "peft":
        return _peft(sd), fmt
    if fmt == "uno":
        return _uno(sd, model_sd_shapes), fmt
    return sd, fmt  # already a ComfyUI-readable format: let ComfyUI try
