import Foundation

/// Lance backbone configuration, decoded from the published checkpoint's `config.json`
/// **as-is** (no added keys — the mlx-community repos are consumed unchanged).
///
/// Defaults mirror `build_text_config()` in the Python reference (`_loader.py:49`).
/// Note: the checkpoint's config may declare `tie_word_embeddings: true`, but the runtime
/// is untied — `lm_head.weight` exists in the safetensors and must be loaded (spec §Backbone).
public struct LanceTextConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float

    /// head_dim = hiddenSize / numAttentionHeads (128 for Lance-3B).
    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// mRoPE section split [temporal, height, width] over headDim; remainder unrotated.
    /// Hardcoded to match Qwen2.5-VL (mlx-vlm `Qwen2RotaryEmbedding`).
    public var mropeSection: [Int] { [16, 24, 24] }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 128_000
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
    }

    /// Load from a checkpoint directory's `config.json` (top-level keys).
    public static func load(from directory: URL) throws -> LanceTextConfig {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        return try JSONDecoder().decode(LanceTextConfig.self, from: data)
    }
}

/// Token-id constants for the understanding pipeline. Resolved from the tokenizer at load
/// where possible; the EOS pair is declared by the checkpoint's `generation_config.json`.
public enum LanceTokens {
    /// `<|im_end|>` — EOS #1 per generation_config.
    public static let imEnd = 151645
    /// `<|endoftext|>` — EOS #2 per generation_config. Honor BOTH stop ids.
    public static let endOfText = 151643
}

/// Per-token modality groups driving the dual-tower expert routing.
/// Mask rule: `group >= cleanVAE` routes to the GEN tower; below routes to UND.
/// x2t understanding only ever sees `.text` / `.vitSemantic` → UND-only.
public enum PositionGroup: Int32, Sendable {
    case text = 0
    case vitSemantic = 1
    case cleanVAE = 2
    case noisyVAE = 3
}

/// Flat safetensors key names for the published Lance LLM checkpoint (no `model.` prefix).
/// One source of truth shared by the loader and the topology tests — keys are the contract
/// with the published artifact and must never drift.
public enum LanceWeightKeys {
    /// `_moe_gen`-suffixed GEN-tower twin of an UND component name.
    public static func genTwin(_ component: String) -> String { component + "_moe_gen" }

    /// All expected keys for one decoder layer (UND + GEN twins).
    /// QK-norm weights are RMSNorm(headDim) — a Lance addition over stock Qwen2.5-VL.
    public static func layerKeys(_ i: Int) -> [String] {
        let p = "layers.\(i)"
        var keys: [String] = []
        for tower in ["", "_moe_gen"] {
            for proj in ["q_proj", "k_proj", "v_proj"] {
                keys.append("\(p).self_attn.\(proj)\(tower).weight")
                keys.append("\(p).self_attn.\(proj)\(tower).bias")
            }
            keys.append("\(p).self_attn.o_proj\(tower).weight")
            keys.append("\(p).self_attn.q_norm\(tower).weight")
            keys.append("\(p).self_attn.k_norm\(tower).weight")
            keys.append("\(p).input_layernorm\(tower).weight")
            keys.append("\(p).post_attention_layernorm\(tower).weight")
            let mlp = tower.isEmpty ? "mlp" : "mlp_moe_gen"
            keys.append("\(p).\(mlp).gate_proj.weight")
            keys.append("\(p).\(mlp).gate_proj.bias")
            keys.append("\(p).\(mlp).up_proj.weight")
            keys.append("\(p).\(mlp).up_proj.bias")
            keys.append("\(p).\(mlp).down_proj.weight")
        }
        return keys
    }

    /// Root-level keys. `lm_head.weight` is present and distinct (untied head).
    public static let rootKeys = [
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm_moe_gen.weight",
    ]
}
