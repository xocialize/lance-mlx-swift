import Foundation
import MLX
import MLXFast
import MLXNN

// Lance dual-tower MoT backbone — ported from lance_mlx/model/lance_llm.py (the
// parity-validated Python reference). mRoPE math adapted from mlx-swift-lm's MLXVLM
// Qwen25VL (MIT; see NOTICE) — Lance applies true 3-axis mRoPE via position ids,
// matching the Python port, not the sequential-RoPE shortcut.

/// Minimal growing KV cache (concat + offset). One per layer.
public final class LanceKVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public var offset: Int { keys?.dim(2) ?? 0 }

    public init() {}

    public func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            self.keys = k
            self.values = v
        }
        return (self.keys!, self.values!)
    }
}

/// 3-axis mRoPE: positionIds (3, B, T) over [temporal, height, width] with section split
/// [16, 24, 24] of headDim/2 (doubled to cover headDim exactly: 32+48+48 = 128).
enum MRoPE {
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, d...], x[.ellipsis, ..<d]], axis: -1)
    }

    /// cos/sin of shape (B, 1, T, headDim), section-interleaved across the t/h/w planes.
    static func cosSin(
        positionIds: MLXArray, headDim: Int, theta: Float, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        let half = headDim / 2
        let invFreq = pow(
            MLXArray(theta), -MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
                / Float(headDim)
        )  // (half,)
        // (3, B, T, 1) * (half,) → (3, B, T, half)
        let freqs = positionIds.asType(.float32).expandedDimensions(axis: -1) * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)  // (3, B, T, headDim)
        var cosT = cos(emb)
        var sinT = sin(emb)
        // E7 root-cause fix: HF/mlx-vlm split cos/sin by the section list REPEATED —
        // [16,24,24,16,24,24] picking planes t,h,w,t,h,w — so rotate-half pairs (j, j+64)
        // stay within one axis (t↔t, h↔h, w↔w). The previous element-doubled [32,48,48]
        // split (inherited from mlx-swift-lm's transcription of NumPy's `mrope_section * 2`,
        // which is LIST REPETITION, as element-wise multiply) paired t-frequencies with
        // h-frequencies — a no-op for pure-text positions (t=h=w) but spatial scrambling at
        // exactly the vision-token positions inside the decoder: content-reading degraded,
        // fluency/EOS/prior answers untouched (the E7 signature).
        let repeated = mropeSection + mropeSection
        var indices: [Int] = []
        var acc = 0
        for s in repeated.dropLast() { acc += s; indices.append(acc) }
        cosT = concatenated(
            split(cosT, indices: indices, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]
        sinT = concatenated(
            split(sinT, indices: indices, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]
        return (cosT, sinT)
    }

    /// Apply to q/k of shape (B, H, T, headDim).
    static func apply(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let qOut = (q * cos) + (rotateHalf(q) * sin)
        let kOut = (k * cos) + (rotateHalf(k) * sin)
        return (qOut, kOut)
    }
}

/// SwiGLU MLP. Bias-free on all three projections — like stock Qwen2.5. (The Lance-3B-bf16
/// checkpoint ships no `gate_proj.bias`/`up_proj.bias`; only the *attention* projections carry
/// biases. An earlier port declared gate/up bias:true and failed the L1 load with 144 missing keys.)
public final class LanceMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Dual-tower attention with QK-norm (Lance additions over stock Qwen2.5-VL) and per-token
/// expert routing. For x2t the gen mask is all-false; pass `genMask: nil` for the UND-only
/// fast path (skips the GEN tower entirely — Python precedent `_und_only_forward`).
public final class LanceMoTAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float
    let mropeSection: [Int]

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    @ModuleInfo(key: "q_proj_moe_gen") var wqGen: Linear
    @ModuleInfo(key: "k_proj_moe_gen") var wkGen: Linear
    @ModuleInfo(key: "v_proj_moe_gen") var wvGen: Linear
    @ModuleInfo(key: "o_proj_moe_gen") var woGen: Linear
    @ModuleInfo(key: "q_norm_moe_gen") var qNormGen: RMSNorm
    @ModuleInfo(key: "k_norm_moe_gen") var kNormGen: RMSNorm

    public init(_ config: LanceTextConfig) {
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.ropeTheta = config.ropeTheta
        self.mropeSection = config.mropeSection

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self._wqGen.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wkGen.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wvGen.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._woGen.wrappedValue = Linear(heads * headDim, dim, bias: false)
        self._qNormGen.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNormGen.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// - Parameters:
    ///   - x: (B, T, D) hidden states (already expert-normed by the layer)
    ///   - positionIds: (3, B, T) t/h/w grid for mRoPE
    ///   - genMask: (B, T, 1) float 1=GEN 0=UND, or nil for the UND-only fast path
    public func callAsFunction(
        _ x: MLXArray, positionIds: MLXArray, genMask: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LanceKVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        func project(_ wq: Linear, _ wk: Linear, _ wv: Linear, _ qn: RMSNorm, _ kn: RMSNorm)
            -> (MLXArray, MLXArray, MLXArray)
        {
            var q = wq(x).reshaped(B, L, heads, headDim)
            var k = wk(x).reshaped(B, L, kvHeads, headDim)
            let v = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            // Lance QK-norm: per-head RMSNorm on headDim, before RoPE.
            q = qn(q).transposed(0, 2, 1, 3)
            k = kn(k).transposed(0, 2, 1, 3)
            return (q, k, v)
        }

        var (queries, keys, values) = project(wq, wk, wv, qNorm, kNorm)
        if let genMask {
            let (qG, kG, vG) = project(wqGen, wkGen, wvGen, qNormGen, kNormGen)
            // (B, T, 1) → (B, 1, T, 1) to broadcast over heads/headDim.
            let m = genMask.transposed(0, 2, 1).expandedDimensions(axis: -1)
            queries = m * qG + (1 - m) * queries
            keys = m * kG + (1 - m) * keys
            values = m * vG + (1 - m) * values
        }

        let (cosT, sinT) = MRoPE.cosSin(
            positionIds: positionIds, headDim: headDim, theta: ropeTheta,
            mropeSection: mropeSection)
        (queries, keys) = MRoPE.apply(q: queries, k: keys, cos: cosT, sin: sinT)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)

        if let genMask {
            return genMask * woGen(output) + (1 - genMask) * wo(output)
        }
        return wo(output)
    }
}

/// One MoT decoder layer: pre-norm residual with every component expert-routed.
public final class LanceMoTLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LanceMoTAttention
    @ModuleInfo(key: "mlp") var mlp: LanceMLP
    @ModuleInfo(key: "mlp_moe_gen") var mlpGen: LanceMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "input_layernorm_moe_gen") var inputNormGen: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm_moe_gen") var postNormGen: RMSNorm

    public init(_ config: LanceTextConfig) {
        self._attention.wrappedValue = LanceMoTAttention(config)
        self._mlp.wrappedValue = LanceMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._mlpGen.wrappedValue = LanceMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._inputNormGen.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postNormGen.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, positionIds: MLXArray, genMask: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: LanceKVCache?
    ) -> MLXArray {
        let normed: MLXArray
        if let genMask {
            normed = genMask * inputNormGen(x) + (1 - genMask) * inputNorm(x)
        } else {
            normed = inputNorm(x)
        }
        let h = x + attention(normed, positionIds: positionIds, genMask: genMask,
                              mask: mask, cache: cache)
        let mlpOut: MLXArray
        if let genMask {
            let normed2 = genMask * postNormGen(h) + (1 - genMask) * postNorm(h)
            mlpOut = genMask * mlpGen(normed2) + (1 - genMask) * mlp(normed2)
        } else {
            mlpOut = mlp(postNorm(h))
        }
        return h + mlpOut
    }
}

/// The Lance backbone: embeddings → 36 MoT layers → expert-routed final norm → untied LM head.
public final class LanceModel: Module {
    public let config: LanceTextConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [LanceMoTLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "norm_moe_gen") var normGen: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: LanceTextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            LanceMoTLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._normGen.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// Forward over pre-built input embeddings (ViT features already merged for x2t).
    /// `positionGroup` (B, T) int32; nil = all-text (UND-only). Returns (B, T, hidden).
    public func callAsFunction(
        inputEmbeddings: MLXArray, positionIds: MLXArray, positionGroup: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, caches: [LanceKVCache]?
    ) -> MLXArray {
        // gen mask: group >= cleanVAE → GEN tower. nil group (or all-UND) takes the fast path.
        var genMask: MLXArray?
        if let positionGroup {
            let isGen = positionGroup .>= MLXArray(PositionGroup.cleanVAE.rawValue)
            if isGen.any().item(Bool.self) {
                genMask = isGen.asType(inputEmbeddings.dtype).expandedDimensions(axis: -1)
            }
        }

        var h = inputEmbeddings
        for (i, layer) in layers.enumerated() {
            h = layer(h, positionIds: positionIds, genMask: genMask, mask: mask,
                      cache: caches?[i])
        }
        if let genMask {
            h = genMask * normGen(h) + (1 - genMask) * norm(h)
        } else {
            h = norm(h)
        }
        return h
    }

    /// Logits for the final position(s) via the untied head.
    public func logits(_ hidden: MLXArray) -> MLXArray { lmHead(hidden) }
}
