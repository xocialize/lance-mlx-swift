// Adapted from mlx-swift-lm (https://github.com/ml-explore/mlx-swift-lm, MIT License)
// MLXVLM/Models/Qwen25VL.swift — the Qwen2.5-VL vision tower, which is fileprivate
// upstream. Adapted for Lance: namespaced as LanceVision, access opened to public.
// See NOTICE for attribution. Lance uses this ViT unchanged (stock Qwen2.5-VL weights
// in the published vit.safetensors).

import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM

public enum LanceVision {

    static public func applyMultimodalRotaryPositionEmbedding(
        _ tensor: MLXArray, freqs: MLXArray
    ) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        let output = (tensor * cos) + (MRoPE.rotateHalf(tensor) * sin)
        return output.asType(tensor.dtype)
    }

    public class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        public init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            self.hiddenSize = contextDimensions * (spatialMergeSize * spatialMergeSize)
            self._layerNormQ.wrappedValue = RMSNorm(dimensions: contextDimensions, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions)
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    public class Attention: Module {

        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dims: Int, numHeads: Int) {
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
            self._proj.wrappedValue = Linear(dims, dims)
        }

        public func callAsFunction(
            _ x: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            let qkv = qkv(x)
            let s = split(qkv, parts: 3, axis: -1)
            var (q, k, v) = (s[0], s[1], s[2])

            q = q.reshaped(sequenceLength, numHeads, -1)
            k = k.reshaped(sequenceLength, numHeads, -1)
            v = v.reshaped(sequenceLength, numHeads, -1)

            q = applyMultimodalRotaryPositionEmbedding(q, freqs: rotaryPositionEmbedding)
            k = applyMultimodalRotaryPositionEmbedding(k, freqs: rotaryPositionEmbedding)

            q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)

            // E6 fix: the upstream copy accepted `attentionMask` and then passed `.none`,
            // so EVERY block ran full attention — but only `fullattBlockIndexes` should;
            // the rest are window-restricted. cosine vs Python went 0.506 → use the mask.
            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .array(attentionMask)
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(output)
        }
    }

    public class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    public class Qwen25VLVisionBlock: Module {

        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self.norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

            self._attention.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numHeads)

            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        }

        public func callAsFunction(
            _ hiddenStates: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
        ) -> MLXArray {
            var hiddenStates =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding
                )
            hiddenStates = hiddenStates + mlp(norm2(hiddenStates))
            return hiddenStates
        }
    }

    public class VisionModel: Module {

        @ModuleInfo(key: "patch_embed") var patchEmbed: LancePatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: LanceVisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
        @ModuleInfo(key: "merger") var patchMerger: PatchMerger

        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullattBlockIndexes: [Int]

        public init(_ config: Qwen25VLConfiguration.VisionConfiguration) {
            self.spatialMergeSize = config.spatialMergeSize
            self.windowSize = config.windowSize
            self.patchSize = config.patchSize
            self.spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            self.fullattBlockIndexes = config.fullattBlockIndexes

            self._patchEmbed.wrappedValue = LancePatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDimensions = config.hiddenSize / config.numHeads
            self._rotaryPositionEmbedding.wrappedValue = LanceVisionRotaryEmbedding(
                dimensions: headDimensions / 2, theta: 10_000)

            self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
                Qwen25VLVisionBlock(config)
            }
            self._patchMerger.wrappedValue = PatchMerger(
                dimensions: config.outHiddenSize, contextDimensions: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize)
        }

        func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIds = [MLXArray]()

            for row in frames {
                let (t, h, w) = row.values

                var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIds = repeated(hposIds, count: w, axis: 1)
                hposIds =
                    hposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIds = repeated(wposIds, count: h, axis: 0)
                wposIds =
                    wposIds
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
                positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIds, axis: 0)
            let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbedFull = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[
                indices]

            return rotaryPositionEmbedFull.reshaped(indices.dim(0), -1)
        }

        func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexId = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let (gridT, gridH, gridW) = frame.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize

                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW)).reshaped(
                    gridT, llmGridH, llmGridW)

                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize

                // Pad the index
                let indexPadded = padded(
                    index,
                    widths: [[0, 0], [0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100)
                )

                // Reshape and transpose
                let indexReshaped = indexPadded.reshaped(
                    gridT,
                    numWindowsH,
                    vitMergerWindowSize,
                    numWindowsW,
                    vitMergerWindowSize
                )

                let indexTransposed = indexReshaped.transposed(0, 1, 3, 2, 4).reshaped(
                    gridT,
                    numWindowsH * numWindowsW,
                    vitMergerWindowSize,
                    vitMergerWindowSize
                )

                // Calculate sequence lengths
                let seqlens = sum(indexTransposed .!= -100, axes: [2, 3]).reshaped(-1)

                // Get valid indices
                let indexFlattened = indexTransposed.flattened()
                let validIndices = indexFlattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map { $0.offset }

                let validValues = indexFlattened[MLXArray(validIndices)]

                // Add to window index
                windowIndex.append(validValues + windowIndexId)

                // Update cumulative sequence lengths
                let cuSeqlensTmp =
                    cumsum(seqlens, axis: 0) * spatialMergeUnit + cuWindowSeqlens.last!
                cuWindowSeqlens.append(contentsOf: cuSeqlensTmp.asArray(Int.self))

                windowIndexId += gridT * llmGridH * llmGridW
            }

            // Concatenate all window indices
            let combinedWindowIndex = concatenated(windowIndex, axis: 0)
            let cuWindowSeqlensArray = MLXArray(cuWindowSeqlens)

            // Get unique values in cuWindowSeqlens
            var seen = Set<Int>()
            var uniqueIndices = [Int]()

            for (i, value) in cuWindowSeqlens.enumerated() {
                if !seen.contains(value) {
                    seen.insert(value)
                    uniqueIndices.append(i)
                }
            }

            let uniqueCuWindowSeqlens = cuWindowSeqlensArray[MLXArray(uniqueIndices)]

            return (combinedWindowIndex, uniqueCuWindowSeqlens)
        }

        /// Block-diagonal boolean mask from cumulative segment boundaries — (1, 1, S, S),
        /// true = may attend. Python realizes this by splitting q/k/v per segment and running
        /// SDPA per window; one masked SDPA is equivalent.
        ///
        /// E6 fix (lance-mlx-swift): the upstream version built this with a subscript-setter
        /// (`mask[0..., a..<b, a..<b] = true`) — a silent no-op in mlx-swift (proven in E6
        /// run #2), yielding an all-false mask; and the Attention discarded it anyway.
        /// Built functionally instead: segment id per position, mask = (seg_i == seg_j).
        func attentionMask(sequenceLength: Int, cuSeqlens: MLXArray) -> MLXArray {
            let bounds = cuSeqlens.asArray(Int.self)
            let positions = MLXArray((0 ..< sequenceLength).map { Int32($0) })  // (S,)
            // Interior boundaries only (drop leading 0 and trailing S).
            let inner = bounds.dropFirst().dropLast().map { Int32($0) }
            let segmentIds: MLXArray
            if inner.isEmpty {
                segmentIds = MLXArray.zeros([sequenceLength]).asType(.int32)
            } else {
                let b = MLXArray(Array(inner)).expandedDimensions(axis: 1)      // (B, 1)
                segmentIds = (positions .>= b).asType(.int32).sum(axis: 0)      // (S,)
            }
            let mask = segmentIds.expandedDimensions(axis: 1)
                .== segmentIds.expandedDimensions(axis: 0)                       // (S, S) bool
            return mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // (1, 1, S, S)
        }

        /// E6 bisection capture (not a Module parameter — plain reference box, ignored by
        /// MLXNN reflection so weight-load verification is unaffected).
        public final class DebugCapture {
            public var stages: [String: MLXArray] = [:]
            public init() {}
        }
        public let debug = DebugCapture()

        public func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            let capture = ProcessInfo.processInfo.environment["LANCE_DEBUG"] == "1"
            var hiddenStates = patchEmbed(hiddenStates)
            if capture { debug.stages["post_patch_embed"] = hiddenStates }
            let rotaryPosEmb = rotaryPositionEmbedding(frames)

            // Get window indices and sequence lengths
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)

            // prepare attention masks
            let seqLen = hiddenStates.dim(0)
            var cuSeqlens = [0]
            for frame in frames {
                let seqLen = frame.h * frame.w
                cuSeqlens.append(
                    contentsOf: Array(repeating: seqLen, count: frame.t).map {
                        cuSeqlens.last! + $0
                    })
            }
            let cuSeqlensArray = MLXArray(cuSeqlens)

            let fullAttentionMask = attentionMask(sequenceLength: seqLen, cuSeqlens: cuSeqlensArray)
            let windowAttentionMask = attentionMask(
                sequenceLength: seqLen, cuSeqlens: cuWindowSeqlens)

            // Reshape and reindex hidden states
            hiddenStates = hiddenStates.reshaped(seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            hiddenStates = hiddenStates[windowIndex, 0..., 0...]
            hiddenStates = hiddenStates.reshaped(seqLen, -1)

            // Reshape and reindex rotary position embeddings
            var rotaryPosEmbReshaped = rotaryPosEmb.reshaped(
                seqLen / spatialMergeUnit, spatialMergeUnit, -1)
            rotaryPosEmbReshaped = rotaryPosEmbReshaped[windowIndex, 0..., 0...]
            rotaryPosEmbReshaped = rotaryPosEmbReshaped.reshaped(seqLen, -1)
            if capture {
                debug.stages["window_index"] = windowIndex
                debug.stages["rotary_pos_emb"] = rotaryPosEmbReshaped
            }

            // Process through blocks
            for (i, block) in blocks.enumerated() {
                // Use full attention for specific blocks, window attention for others
                let attentionMask =
                    fullattBlockIndexes.contains(i) ? fullAttentionMask : windowAttentionMask

                if capture {
                    // Fine ladder (run #8 plan): every block captured; the crater span
                    // (16–23) additionally split into post-attention vs post-MLP. The block
                    // is computed via its own submodules in exactly its residual order, so
                    // the split rows are the block's true intermediates, not a recompute.
                    let attnOut = block.attention(
                        block.norm1(hiddenStates),
                        attentionMask: attentionMask,
                        rotaryPositionEmbedding: rotaryPosEmbReshaped)
                    let postAttn = hiddenStates + attnOut
                    if (16...23).contains(i) {
                        debug.stages[String(format: "post_attn%02d", i)] = postAttn
                    }
                    hiddenStates = postAttn + block.mlp(block.norm2(postAttn))
                    debug.stages[String(format: "post_block%02d", i)] = hiddenStates
                } else {
                    hiddenStates = block(
                        hiddenStates,
                        attentionMask: attentionMask,
                        rotaryPositionEmbedding: rotaryPosEmbReshaped
                    )
                }
            }
            if capture {
                debug.stages["pre_merger"] = hiddenStates
                // Merger internals isolated: RMSNorm, then the first MLP projection.
                let normed = patchMerger.layerNormQ(hiddenStates)
                debug.stages["merger_post_norm"] = normed
                debug.stages["merger_post_mlp0"] =
                    patchMerger.mlp.0(normed.reshaped(-1, patchMerger.hiddenSize))
            }

            // Apply patch merger
            hiddenStates = patchMerger(hiddenStates)

            // Reorder back to original sequence
            let reverseIndices = argSort(windowIndex, axis: 0)
            hiddenStates = hiddenStates[reverseIndices, 0...]

            return hiddenStates
        }

        private func isMLXWeight(_ array: MLXArray) -> Bool {
            if array.ndim != 4, array.ndim != 5 {
                return false
            }

            if array.dim(-1) == 3 {
                return true
            }

            let (outChannels, kH, kW) = (array.dim(1), array.dim(2), array.dim(3))
            return outChannels >= kH && outChannels >= kW && kH == kW
        }

        public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_id") {
                    // Remove unused position_ids
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    // PyTorch conv2d weight tensors have shape:
                    //   [B, out_channels, in_channels, kH, KW]
                    // MLX conv2d expects the weight be of shape:
                    //   [B, out_channels, kH, KW, in_channels]
                    if isMLXWeight(v) {
                        sanitizedWeights[k] = v
                    } else {
                        sanitizedWeights[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}
