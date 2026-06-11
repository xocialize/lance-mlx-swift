#!/usr/bin/env python3
"""Dump the Python reference ViT features for one Lance L1 oracle case.

E6 branch-2 tool: if the Swift ablation shows vision IS attended but answers stay
wrong, the next check is ViT semantic parity. Run this from the Python port repo:

    cd /Volumes/DEV_ARCHIVE/lance-mlx
    uv run python /Users/dustinnielson/Development/MLXEngine/lance-mlx-swift/tools/dump_vit_reference.py \
        --case 01 --out /tmp/lance_vit_ref_case01.safetensors

Then compare against the Swift side: with LANCE_VIT_DUMP=<path> set, the Swift
runner writes its imageFeatures for the same case next to the reference; cosine
< 0.99 on the flattened features ⇒ the defect is in preprocess/patchify/sanitize.
Saves: pixel_values (pre-ViT, post-preprocess) and image_features (post-merger),
so the two stages can be compared independently.
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
        lance_weights=str(snapshot),
        vit_weights=str(snapshot / "vit.safetensors"),
    )

    inputs = pipe.processor(images=image, text="<|vision_start|><|image_pad|><|vision_end|>x",
                            return_tensors="mlx")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    features = pipe.vision_model(
        pixel_values.astype(mx.bfloat16), grid_thw
    )

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
