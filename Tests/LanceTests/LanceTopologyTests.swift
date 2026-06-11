import Foundation
import Testing
import MLX
@testable import Lance

// Topology check: every key in the pinned checkpoint contract must exist in the module
// tree, so load_weights against the published safetensors can never silently miss a
// parameter. Module construction touches MLX stream init, which needs the metallib —
// **xcodebuild-only** (the SPM CLI can't load it; same boundary as RoPEConventionTests
// in mel-roformer). Run via the MLXEngine.xcworkspace test action.

@Suite(.enabled(if: ProcessInfo.processInfo.environment["LANCE_METAL_TESTS"] == "1"))
struct LanceTopologyTests {
    @Test func moduleTreeMatchesCheckpointKeyContract() throws {
        let json = """
        {"hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
         "vocab_size": 100}
        """
        let config = try JSONDecoder().decode(LanceTextConfig.self, from: Data(json.utf8))
        let model = LanceModel(config)

        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        var expected = LanceWeightKeys.rootKeys
        for i in 0..<config.numHiddenLayers { expected += LanceWeightKeys.layerKeys(i) }

        let missing = expected.filter { !moduleKeys.contains($0) }
        #expect(missing.isEmpty, "module tree missing checkpoint keys: \(missing.prefix(6))")

        // And nothing unexpected beyond the contract (rotary tables etc. are not parameters).
        let extra = moduleKeys.subtracting(expected)
        #expect(extra.isEmpty, "module has parameters absent from the checkpoint: \(extra.prefix(6))")
    }
}
