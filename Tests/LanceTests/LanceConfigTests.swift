import Foundation
import Testing
@testable import Lance

// Offline topology/config tests — no weights, no Metal. The key-name contract with the
// published mlx-community checkpoint is pinned here so it can never silently drift.

@Suite struct LanceConfigTests {
    @Test func configDecodesLance3BShape() throws {
        // Mirror of the published Lance-3B-bf16 config.json (relevant top-level keys).
        let json = """
        {"hidden_size": 2048, "num_hidden_layers": 36, "intermediate_size": 11008,
         "num_attention_heads": 16, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
         "vocab_size": 151646}
        """
        let config = try JSONDecoder().decode(LanceTextConfig.self, from: Data(json.utf8))
        #expect(config.hiddenSize == 2048)
        #expect(config.numHiddenLayers == 36)
        #expect(config.headDim == 128)
        #expect(config.numKeyValueHeads == 2)
        #expect(config.ropeTheta == 1_000_000)        // default when absent
        #expect(config.maxPositionEmbeddings == 128_000)
        #expect(config.mropeSection == [16, 24, 24])
    }

    @Test func eosPairMatchesGenerationConfig() {
        #expect(LanceTokens.imEnd == 151645)
        #expect(LanceTokens.endOfText == 151643)
    }

    @Test func routingMaskBoundaryIsCleanVAE() {
        // x2t groups route UND; VAE groups route GEN.
        #expect(PositionGroup.text.rawValue < PositionGroup.cleanVAE.rawValue)
        #expect(PositionGroup.vitSemantic.rawValue < PositionGroup.cleanVAE.rawValue)
        #expect(PositionGroup.noisyVAE.rawValue >= PositionGroup.cleanVAE.rawValue)
    }

    @Test func layerKeyContractMatchesPublishedCheckpoint() {
        let keys = LanceWeightKeys.layerKeys(0)
        // Spot-check exact strings against the published safetensors layout.
        #expect(keys.contains("layers.0.self_attn.q_proj.weight"))
        #expect(keys.contains("layers.0.self_attn.q_proj_moe_gen.bias"))
        #expect(keys.contains("layers.0.self_attn.k_norm.weight"))
        #expect(keys.contains("layers.0.self_attn.q_norm_moe_gen.weight"))
        #expect(keys.contains("layers.0.input_layernorm_moe_gen.weight"))
        #expect(keys.contains("layers.0.mlp.gate_proj.bias"))
        #expect(keys.contains("layers.0.mlp_moe_gen.down_proj.weight"))
        // No o_proj bias, no down_proj bias (bias only on q/k/v and gate/up).
        #expect(!keys.contains("layers.0.self_attn.o_proj.bias"))
        #expect(!keys.contains("layers.0.mlp.down_proj.bias"))
        // 2 towers × (q/k/v w+b = 6, o_proj w = 1, 2 qk-norms, 2 layer norms, 5 mlp) = 2 × 16 = 32
        #expect(keys.count == 32)
        #expect(LanceWeightKeys.rootKeys.contains("lm_head.weight")) // untied head
    }
}
