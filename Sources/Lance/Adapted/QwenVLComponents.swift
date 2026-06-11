// Adapted from mlx-swift-lm (https://github.com/ml-explore/mlx-swift-lm, MIT License)
// MLXVLM/Models/QwenVL.swift — VisionRotaryEmbedding + PatchEmbed, whose initializers
// are internal upstream. See NOTICE for attribution.

import Foundation
import MLX
import MLXNN

public class LanceVisionRotaryEmbedding {
    let dimensions: Int
    let theta: Float
    let inverseFreq: MLXArray

    public init(dimensions: Int, theta: Float) {
        self.dimensions = dimensions
        self.theta = theta
        let p = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
        self.inverseFreq = 1.0 / pow(theta, p)
    }

    public func callAsFunction(sequenceLength: Int) -> MLXArray {
        let seq = MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype)
        let freqs = outer(seq, inverseFreq)
        return freqs
    }
}

public class LancePatchEmbed: Module, UnaryLayer {
    @ModuleInfo var proj: Conv3d

    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let outputDimensions: Int

    public convenience init(
        patchSize: Int, temporalPatchSize: Int, inChannels: Int, hiddenSize: Int
    ) {
        self.init(
            patchSize: patchSize, temporalPatchSize: temporalPatchSize,
            inChannels: inChannels, outputDimensions: hiddenSize)
    }

    public init(patchSize: Int, temporalPatchSize: Int, inChannels: Int, outputDimensions: Int) {
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.inChannels = inChannels
        self.outputDimensions = outputDimensions

        let kernelSize = IntOrTriple([temporalPatchSize, patchSize, patchSize])
        self._proj.wrappedValue = Conv3d(
            inputChannels: inChannels,
            outputChannels: outputDimensions,
            kernelSize: kernelSize,
            stride: kernelSize,
            bias: false
        )
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates = hiddenStates.reshaped(
            -1, inChannels, temporalPatchSize, patchSize, patchSize
        ).movedAxis(source: 1, destination: 4)

        hiddenStates = proj(hiddenStates)
        hiddenStates = hiddenStates.reshaped(-1, outputDimensions)
        return hiddenStates
    }
}
