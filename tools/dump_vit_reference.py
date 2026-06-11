#!/usr/bin/env python3
"""Dump the Python reference ViT tensors for one Lance L1 oracle case.

E6 branch-2 tool: the Swift ablation proved vision is attended but semantically wrong,
so this dumps the reference pixel_values (post-preprocess, pre-ViT) and image_features
(post-merger) for direct tensor comparison against the Swift pipeline:

    cd /Volumes/DEV_ARCHIVE/lance-mlx
    uv run python /Users/dustinnielson/Development/MLXEngine/lance-mlx-swift/tools/dump_vit_reference.py \
        --case 01 --out /Users/dustinnielson/Development/MLXEngine/lance-vit-ref-case01.safetensors

The Swift runner (LANCE_DEBUG=1) auto-compares against that well-known path when the
input grid matches. pixel_values mismatch -> preprocess/patchify; pixel match but
feature mismatch -> ViT forward / sanitize transpose.
"""

import argparse
from pathlib import Path

import mlx.core as mx
from PIL import Image

from lance_mlx.pipeline.understanding import UnderstandingPipeline

REPO = Path("/Volumes/DEV_ARCHIVE/lance-mlx")
SNAPSHOT_GLOB = "models--mlx-community--Lance-3B-bf16/snapshots/*"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="Lance-3B-bf16 snapshot dir")
    args = ap.parse_args()

    if args.weights:
        snapshot = Path(args.weights)
    else:
        cache = Path.home() / ".cache/huggingface/hub"
        snapshot = sorted(cache.glob(SNAPSHOT_GLOB))[-1]

    image_path = (
        REPO / "tests/fixtures/images" / f"image-understanding-case-{args.case}.png"
    )
    image = Image.open(image_path).convert("RGB")

    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=str(snapshot),
        vit_safetensors=str(snapshot / "vit.safetensors"),
    )

    text = "<|vision_start|><|image_pad|><|vision_end|>x"
    inputs = pipe.processor(images=image, text=text, return_tensors="mlx")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]

    vit_dtype = pipe.vision_model.patch_embed.proj.weight.dtype
    features = pipe.vision_model(pixel_values.astype(vit_dtype), grid_thw)

    mx.save_safetensors(
        args.out,
        {
            "pixel_values": pixel_values.astype(mx.float32),
            "image_features": features.astype(mx.float32),
            "grid_thw": grid_thw.astype(mx.int32),
        },
    )
    print(f"case {args.case}: pixel_values {pixel_values.shape} "
          f"features {features.shape} grid {grid_thw.tolist()} -> {args.out}")


if __name__ == "__main__":
    main()
