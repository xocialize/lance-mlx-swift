# lance-mlx-swift

Swift/MLX port of **Lance** (ByteDance Intelligent Creation Lab's unified multimodal model —
[paper](https://arxiv.org/abs/2605.18678)) for Apple Silicon, ported from our
production-validated Python port [xocialize/lance-mlx](https://github.com/xocialize/lance-mlx).
Consumes the published [mlx-community Lance checkpoints](https://huggingface.co/collections/mlx-community/lance-mlx-6a0f3cd5648a74f8283fc8a4)
**exactly as published** — no conversion, no re-upload.

> **Status: L1 (image understanding) — code complete, parity validation pending.**
> The dual-tower MoT backbone, weight loader, Qwen2.5-VL vision tower, and the
> `x2t_image` VQA pipeline build and are key-contract-tested; the 6-case oracle parity
> run against the Python reference has not yet been executed (requires Xcode/Metal).
> Generation (t2i/t2v/editing) is not started — see `PORTING-SPEC.md` for the phase plan.

## L1 usage (image VQA)

```swift
import Lance

let pipeline = try await LanceUnderstanding.load(
    directory: lanceSnapshotURL)  // mlx-community/Lance-3B-bf16 snapshot
let answer = try pipeline.generate(
    image: ciImage, question: "What is shown in this image?")
```

- `LanceModel` — 36-layer dual-tower MoT (UND/GEN experts, QK-norm, 3-axis mRoPE,
  per-token routing with an UND-only fast path).
- `LanceLoader` — two-way key verification against the published safetensors; refuses
  partial loads.
- `LanceUnderstanding` — Lance-template VQA: smart-resize → ViT → feature merge →
  3D position ids → greedy decode (dual EOS).

`Sources/Lance/Adapted/` contains the Qwen2.5-VL vision tower adapted from
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (MIT — see NOTICE), which is
`fileprivate` upstream.

## License

Apache-2.0 (this port). Lance weights: Apache-2.0 (ByteDance). Qwen2.5-VL / Wan2.2 VAE:
Apache-2.0 (Alibaba). See `NOTICE`.
