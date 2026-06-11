import Foundation
import MLX
import MLXNN

public enum LanceLoaderError: Error, Equatable {
    /// The checkpoint left module parameters unfilled — wrong/incompatible weights.
    /// A partial load would emit garbage with no other symptom; refuse instead.
    case missingKeys(count: Int, examples: [String])
}

/// Loads the published Lance LLM checkpoint (`model.safetensors`, flat keys, mixed
/// bf16/f32 dtypes preserved) into a `LanceModel`. The published artifact is the
/// contract — keys are verified both ways against the module tree.
public enum LanceLoader {
    public struct LoadResult {
        public let model: LanceModel
        public let config: LanceTextConfig
        /// Checkpoint keys the module didn't consume (should be empty for Lance-3B).
        public let unusedKeys: [String]
    }

    public static func loadModel(directory: URL) throws -> LoadResult {
        let config = try LanceTextConfig.load(from: directory)
        let model = LanceModel(config)

        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("model.safetensors"))

        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)

        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw LanceLoaderError.missingKeys(
                count: missing.count, examples: Array(missing.prefix(5)))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()

        // Load only module-known keys; dtypes (bf16 weights, f32 norms) pass through as-is.
        let consumed = weights.filter { moduleKeys.contains($0.key) }
        model.update(parameters: ModuleParameters.unflattened(consumed))
        eval(model)

        return LoadResult(model: model, config: config, unusedKeys: unused)
    }
}
