# lance-mlx-swift — L1 porting spec (x2t_image / understanding path)

Distilled from the Python reference at `/Volumes/DEV_ARCHIVE/lance-mlx` (production-validated)
on 2026-06-10. **Hard rule: consume the published `mlx-community/Lance-3B-bf16` checkpoint
exactly as-is** (705+ downloads; no config keys added, no re-uploads, no renames). Files in the
snapshot: `config.json`, `llm_config.json`, `generation_config.json`, `model.safetensors` (LLM),
`vit.safetensors`, `vae.safetensors` (unused in L1), `tokenizer.json`, `vocab.json`.

## Backbone (LanceModel)

Qwen2.5-VL-3B-derived dual-tower MoT. Config from `config.json` (`build_text_config`,
`_loader.py:49`): hidden 2048 · 36 layers · 16 heads / 2 KV heads (GQA 8:1) · head_dim 128 ·
FFN 11008 (SwiGLU, gate/up bias **true**, down bias false; attention q/k/v bias **true**, o
bias false) · vocab 151646 · rms_norm_eps 1e-6 · rope_theta 1e6 · mrope_section [16,24,24]
(t/h/w of head_dim 128, remainder 64 unrotated) · **untied lm_head** (config may say tied —
runtime overrides to untied; `lm_head.weight` exists in the safetensors).

Weight keys are FLAT (no `model.` prefix). Per layer `i` (UND + `_moe_gen` GEN twin of each):
`layers.{i}.self_attn.{q,k,v,o}_proj[.weight/.bias]`, `…{q,k}_norm.weight` (RMSNorm(128) —
**QK-norm, Lance addition over stock Qwen2.5-VL**), `layers.{i}.input_layernorm.weight`,
`layers.{i}.post_attention_layernorm.weight`, `layers.{i}.mlp.{gate,up,down}_proj.weight`
(+ gate/up `.bias`). GEN twin = same key + `_moe_gen` suffix (on the component: e.g.
`q_proj_moe_gen`, `mlp_moe_gen.gate_proj`, `input_layernorm_moe_gen`). Root:
`embed_tokens.weight`, `lm_head.weight`, `norm.weight`, `norm_moe_gen.weight`.

**Dtype rule** (from `02_convert.py` KEEP_F32_PATTERNS): all norm weights (incl. QK-norms and
final norms) are f32 in the checkpoint even when the rest is bf16 — load without forcing dtype.

## Routing (x2t = UND-only)

`position_group` per token: TEXT=0, VIT_SEMANTIC=1, CLEAN_VAE=2, NOISY_VAE=3. Expert mask
= `group >= 2` → GEN. For x2t both groups are 0/1 → **gen_mask all false; only the UND tower
executes**. Python pattern is non-short-circuit `mx.where(genMask, genPath(x), undPath(x))` at:
pre-attn norm, q/k/v/o (+ QK-norms), post-attn norm, MLP, final norm. L1 Swift: implement the
routed layer faithfully but support a `undOnly` fast path + optional GEN-weight skip at load
(Python precedent: `defer_gen_tower` / `_und_only_forward`). MaPE (temporal re-anchor 1000/2000
for gen modalities) is **not exercised** in x2t — port it with the layer (it's tiny, `mape.py`)
but it's a no-op for L1.

## ViT + processing (reuse MLXVLM where public)

ViT = stock Qwen2.5-VL VisionModel from `vit.safetensors` (mlx-vlm `VisionModel.sanitize`
transposes `patch_embed.proj.weight`). Config from `config.json`'s `vision_config`
(HF `in_chans` → `in_channels` rename). mlx-swift-lm (rev in workspace, mlx-swift ≥0.31.3):
- REUSE (public): `QwenVL.PatchEmbed`, `QwenVL.VisionRotaryEmbedding`, `QwenVL.targetSize`,
  `QwenVL.patchify`, `Qwen25VLProcessor(+Configuration)`, `KVCache` protocol (MLXLMCommon).
- COPY-ADAPT (fileprivate in `MLXVLM/Models/Qwen25VL.swift`): `Vision.VisionModel` (383–622),
  `Vision.PatchMerger` (258–280), `Vision.Attention` (282–331), `Language.Attention` (40–118,
  base for LanceMoTAttention — add QK-norms + GEN twins), `Language.MLP` (120–134),
  decoder layer (136–162). MIT — attribute in NOTICE.

## Understanding pipeline (x2t_image)

Template (Lance-style, default — images are 1-frame videos):
```
<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n
```
`instruction` default: "Look at the image carefully and answer the question."
**After tokenization, image_pad token ids are substituted with video_pad ids** (Lance training
convention). Token ids from tokenizer: image_pad, video_pad, vision_start; EOS = **both**
151645 `<|im_end|>` and 151643 `<|endoftext|>` (generation_config declares both — honor both).

Flow: processor(image, text) → pixel_values + image_grid_thw → ViT forward → merge ViT
features into text embeds at image-pad positions (cumsum-slotting, `understanding.py:169`) →
3D mRoPE position ids (`_compute_position_ids`, `understanding.py:37` — adapted from mlx-vlm
`get_rope_index`; grid t,h,w with h,w divided by spatial_merge_size) → **greedy** AR decode
(argmax, no temperature/top-k; repetition penalty 1.05 exists in reference Python torch but is
NOT implemented in lance-mlx greedy — match lance-mlx), max_new_tokens 256, KV cache standard.
Known Python bug, do NOT replicate: `understanding.py:59` references undefined `video_grid_thw`.

x2t_video (L4): even frame count required (temporal_patch_size 2), LANCZOS resize to 224²,
video_pad directly, frames linearly sampled.

## Parity gate

6 oracle VQA cases: `prompts/understanding_eval.json` + fixtures
`tests/fixtures/x2t_image/u0*/input.png` + expected answers in
`tests/fixtures/results/x2t_image_sample_*/result.json`. Bar = **content-correct** (the Python
port itself is ~95% functional parity vs PyTorch — style differs, content matches). Numerics:
short sequences → bf16 fine; `_rope_fp32`/`_attention_fp32` escape hatches exist in Python for
long-sequence generation only (L2+ concern).

## Later phases (do not build in L1, keep seams)

L2 t2i: GEN tower live, MaPE active, flow head (`llm2vae`, `time_embedder`), Euler then
DPM-Solver++(2M), Wan2.2 48-ch VAE decode (the long pole), PrefixKVCache (frozen text prefix
across denoise steps), n_lat ≤ 16,128 envelope. L3 image_edit (+ contract 1.2.0 imageEdit/
videoEdit). L4 video checkpoint → LanceVideoPackage.
