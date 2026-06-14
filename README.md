# lance-mlx-swift

> # ⚠️ ARCHIVAL — DROPPED (2026-06-12)
>
> **Lance was dropped on 2026-06-12.** Its non-generation components underperformed and were
> holding the project back, so the port is no longer active development. This package is kept
> **for forensics only** (the `v0.1.0` tag is retained); everything below is **historical**.
>
> - **Salvage:** the verified Qwen2.5-VL piece lives on as the separate package
>   **`qwen25vl-mlx-swift`** (serving `imageAnalysis` / `videoAnalysis`).
> - **Generation replacement:** the t2i/t2v direction moved to **Bernini-R**
>   (`bernini-r-mlx-swift`).

---

Swift/MLX port of **Lance** (ByteDance Intelligent Creation Lab's unified multimodal model —
[paper](https://arxiv.org/abs/2605.18678)) for Apple Silicon, ported from our
production-validated Python port [xocialize/lance-mlx](https://github.com/xocialize/lance-mlx).
Consumes the published [mlx-community Lance checkpoints](https://huggingface.co/collections/mlx-community/lance-mlx-6a0f3cd5648a74f8283fc8a4)
**exactly as published** — no conversion, no re-upload.

> **Historical status (at drop): L1 (image understanding) — code complete, parity validation
> pending.** The dual-tower MoT backbone, weight loader, Qwen2.5-VL vision tower, and the
> `x2t_image` VQA pipeline built and were key-contract-tested; the 6-case oracle parity
> run against the Python reference was never executed. The generation (t2i/t2v/editing) phase
> plan in `PORTING-SPEC.md` was never started and is **abandoned** — generation moved to
> `bernini-r-mlx-swift`.

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
