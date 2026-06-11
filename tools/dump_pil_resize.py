#!/usr/bin/env python3
"""Dump PIL-bicubic-resized RGB bytes for the 6 L1 fixtures (E6 resampler fix gate).

The HF Qwen2.5-VL processor resizes with PIL BICUBIC on uint8 RGB before rescale/normalize.
This emits, per case: the smart-resize target and the PIL-resized raw RGB bytes, so the Swift
PIL-exact resampler can be verified OFFLINE (byte compare, no inference, no Metal).

    cd /Volumes/DEV_ARCHIVE/lance-mlx
    uv run python <this file> --out-dir /Volumes/DEV_ARCHIVE/lance-pil-resize

Writes case<NN>.bin (raw RGB8, H*W*3 bytes, row-major) + case<NN>.json (dims).
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image

REPO = Path("/Volumes/DEV_ARCHIVE/lance-mlx")
PATCH = 14
MERGE = 2
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280


def smart_resize(height: int, width: int, factor: int) -> tuple[int, int]:
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // factor) * factor, (w_bar // factor) * factor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 7):
        case = f"{i:02d}"
        path = REPO / "tests/fixtures/images" / f"image-understanding-case-{case}.png"
        im = Image.open(path).convert("RGB")
        h_bar, w_bar = smart_resize(im.height, im.width, PATCH * MERGE)
        resized = im.resize((w_bar, h_bar), Image.BICUBIC)
        raw = resized.tobytes()  # RGB8 row-major
        (out_dir / f"case{case}.bin").write_bytes(raw)
        (out_dir / f"case{case}.json").write_text(json.dumps({
            "source": str(path), "src_w": im.width, "src_h": im.height,
            "dst_w": w_bar, "dst_h": h_bar, "bytes": len(raw),
        }))
        print(f"case {case}: {im.width}x{im.height} -> {w_bar}x{h_bar} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
