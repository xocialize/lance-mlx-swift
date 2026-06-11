#!/usr/bin/env python3
"""Dump Python-reference ViT tensors + intermediate stages for Lance L1 oracle cases.

E6 closer-2 (post run #6, residual cosine 0.9754): emits per-stage intermediates so the
Swift side can bisect the first diverging op. Stages mirror VisionModel.__call__ exactly:
window_index / rotary_pos_emb (exact-comparable), post_patch_embed (pre-reorder),
post_block0 / pre_merger (reordered space), image_features (final), pixel_values, grid_thw.

    cd /Volumes/DEV_ARCHIVE/lance-mlx
    uv run python <this file> --all \
        --weights /Volumes/DEV_ARCHIVE/weights/lance-mlx-models/Lance-3B-bf16 \
        --out-dir /Volumes/DEV_ARCHIVE

Writes lance-vit-ref-case<NN>.safetensors per case (the harness staging location).
"""

import argparse
from pathlib import Path

import mlx.core as mx
from PIL import Image

from lance_mlx.pipeline.understanding import UnderstandingPipeline

REPO = Path("/Volumes/DEV_ARCHIVE/lance-mlx")


def staged_forward(vm, pixel_values, grid_thw):
    """VisionModel.__call__ with stage captures (mirrors mlx_vlm qwen2_5_vl/vision.py)."""
    stages = {}
    vit_dtype = vm.patch_embed.proj.weight.dtype
    h = vm.patch_embed(pixel_values.astype(vit_dtype))
    stages["post_patch_embed"] = h  # pre-reorder

    rotary = vm.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = vm.get_window_index(grid_thw)

    seen = set()
    idx = []
    for i, x in enumerate(cu_window_seqlens):
        if x not in seen:
            seen.add(x)
            idx.append(i)
    idx = mx.array(idx, dtype=mx.int32)
    cu_window_seqlens = cu_window_seqlens[idx]

    seq_len, _ = h.shape
    smu = vm.spatial_merge_unit
    h = h.reshape(seq_len // smu, smu, -1)[window_index, :, :].reshape(seq_len, -1)
    rotary = rotary.reshape(seq_len // smu, smu, -1)[window_index, :, :].reshape(seq_len, -1)
    stages["window_index"] = window_index
    stages["rotary_pos_emb"] = rotary
    stages["cu_window_seqlens"] = cu_window_seqlens

    cu_seqlens = []
    for i in range(grid_thw.shape[0]):
        s = grid_thw[i, 1] * grid_thw[i, 2]
        cu_seqlens.append(mx.repeat(s, grid_thw[i, 0]))
    cu_seqlens = mx.cumsum(mx.concatenate(cu_seqlens).astype(mx.int32), axis=0)
    cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

    for layer_num, blk in enumerate(vm.blocks):
        cu = cu_seqlens if layer_num in vm.fullatt_block_indexes else cu_window_seqlens
        h = blk(h, cu_seqlens=cu, rotary_pos_emb=rotary)
        if layer_num == 0:
            stages["post_block0"] = h
    stages["pre_merger"] = h

    h = vm.merger(h)
    reverse = mx.argsort(window_index, axis=0)
    stages["image_features"] = h[reverse, :]
    return stages


def dump_case(pipe, case: str, out_dir: Path) -> None:
    image_path = REPO / "tests/fixtures/images" / f"image-understanding-case-{case}.png"
    image = Image.open(image_path).convert("RGB")
    text = "<|vision_start|><|image_pad|><|vision_end|>x"
    inputs = pipe.processor(images=image, text=text, return_tensors="mlx")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    stages = staged_forward(pipe.vision_model, pixel_values, grid_thw)
    payload = {
        "pixel_values": pixel_values.astype(mx.float32),
        "grid_thw": grid_thw.astype(mx.int32),
    }
    for k, v in stages.items():
        payload[k] = v.astype(mx.int32 if "index" in k or "seqlens" in k else mx.float32)

    out = out_dir / f"lance-vit-ref-case{case}.safetensors"
    mx.save_safetensors(str(out), payload)
    print(f"case {case}: grid {grid_thw.tolist()} features "
          f"{stages['image_features'].shape} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="01")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", default="/Volumes/DEV_ARCHIVE")
    args = ap.parse_args()

    snapshot = Path(args.weights)
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=str(snapshot),
        vit_safetensors=str(snapshot / "vit.safetensors"),
    )
    cases = [f"{i:02d}" for i in range(1, 7)] if args.all else [args.case]
    for case in cases:
        dump_case(pipe, case, Path(args.out_dir))


if __name__ == "__main__":
    main()
