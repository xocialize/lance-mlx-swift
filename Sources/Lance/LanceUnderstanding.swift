import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

public enum LanceError: Error {
    case imageProcessing(String)
    case missingToken(String)
    case visionConfig(String)
}

/// Image preprocessing for the Lance ViT — smart-resize + patchify math adapted from
/// mlx-swift-lm's QwenVL helpers (internal upstream; MIT, see NOTICE). Constants are the
/// stock Qwen2.5-VL processor values (the Python port fetches the processor from
/// Qwen/Qwen2.5-VL-3B-Instruct; the Lance checkpoint ships no preprocessor config).
public enum LanceImageProcessing {
    public static let imageMean: [CGFloat] = [0.48145466, 0.4578275, 0.40821073]
    public static let imageStd: [CGFloat] = [0.26862954, 0.26130258, 0.27577711]
    public static let patchSize = 14
    public static let mergeSize = 2
    public static let temporalPatchSize = 2
    public static let minPixels = 3136          // 56 × 56
    public static let maxPixels = 1_003_520     // 28 × 28 × 1280

    /// Qwen2.5-VL smart resize: round to factor multiples, scale into the pixel budget.
    public static func targetSize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height >= factor, width >= factor else {
            throw LanceError.imageProcessing(
                "image \(width)×\(height) smaller than patch factor \(factor)")
        }
        guard max(height, width) / min(height, width) <= 200 else {
            throw LanceError.imageProcessing("aspect ratio over 200:1")
        }
        var hBar = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var wBar = max(factor, Int(round(Float(width) / Float(factor))) * factor)
        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(ceil(Float(height) * beta / Float(factor))) * factor
            wBar = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor
        guard hBar > 0, wBar > 0 else {
            throw LanceError.imageProcessing("invalid target \(wBar)×\(hBar)")
        }
        return (hBar, wBar)
    }

    /// Patchify (C,H,W)-stacked frames into the ViT's flattened layout. Exact rearrangement
    /// from QwenVL.patchify — the merge-window interleave must match the published weights.
    public static func patchify(images: [MLXArray]) throws -> (MLXArray, THW) {
        guard let first = images.first else {
            throw LanceError.imageProcessing("no frames")
        }
        let resizedHeight = first.dim(-2)
        let resizedWidth = first.dim(-1)
        var patches = concatenated(images)

        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let lastPatch = patches[-1, .ellipsis]
            let repeated = tiled(lastPatch, repetitions: [temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, repeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT, temporalPatchSize, channel,
            gridH / mergeSize, mergeSize, patchSize,
            gridW / mergeSize, mergeSize, patchSize)
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
        let flattened = patches.reshaped(
            gridT * gridH * gridW,
            channel * temporalPatchSize * patchSize * patchSize)
        return (flattened, THW(gridT, gridH, gridW))
    }

    /// CIImage → normalized (1, C, H, W) float array at the smart-resized resolution.
    public static func preprocess(image: CIImage) throws -> (MLXArray, THW) {
        let h = Int(image.extent.height)
        let w = Int(image.extent.width)
        let (targetH, targetW) = try targetSize(
            height: h, width: w, factor: patchSize * mergeSize,
            minPixels: minPixels, maxPixels: maxPixels)

        let processed = MediaProcessing.apply(image, processing: nil)
        let resized = MediaProcessing.resampleBicubic(
            processed, to: CGSize(width: targetW, height: targetH))
        let normalized = MediaProcessing.normalize(
            MediaProcessing.inSRGBToneCurveSpace(resized),
            mean: (imageMean[0], imageMean[1], imageMean[2]),
            std: (imageStd[0], imageStd[1], imageStd[2]))
        // (1, C, H, W)
        let array = MediaProcessing.asMLXArray(normalized)
        return try patchify(images: [array])
    }
}

/// x2t_image understanding — port of `lance_mlx/pipeline/understanding.py`.
/// Lance convention: images are 1-frame videos — the template uses `<|image_pad|>` but the
/// padded token ids are substituted to `<|video_pad|>` post-tokenization. Greedy decode,
/// dual EOS (151645 + 151643).
public final class LanceUnderstanding {
    public let model: LanceModel
    public let vision: LanceVision.VisionModel
    public let tokenizer: any Tokenizers.Tokenizer
    public let spatialMergeSize: Int

    let imagePadId: Int
    let videoPadId: Int
    let visionStartId: Int

    public static let defaultInstruction =
        "Look at the image carefully and answer the question."

    public init(
        model: LanceModel, vision: LanceVision.VisionModel, tokenizer: any Tokenizers.Tokenizer,
        spatialMergeSize: Int
    ) throws {
        self.model = model
        self.vision = vision
        self.tokenizer = tokenizer
        self.spatialMergeSize = spatialMergeSize
        guard let imagePad = tokenizer.convertTokenToId("<|image_pad|>"),
              let videoPad = tokenizer.convertTokenToId("<|video_pad|>"),
              let visionStart = tokenizer.convertTokenToId("<|vision_start|>")
        else { throw LanceError.missingToken("vision special tokens") }
        self.imagePadId = imagePad
        self.videoPadId = videoPad
        self.visionStartId = visionStart
    }

    /// Load everything from a published checkpoint snapshot (`config.json`, `model.safetensors`,
    /// `vit.safetensors`) + the stock Qwen2.5-VL tokenizer (the checkpoint ships none —
    /// matches the Python port fetching the processor from Qwen/Qwen2.5-VL-3B-Instruct).
    public static func load(
        directory: URL, tokenizerSource: String = "Qwen/Qwen2.5-VL-3B-Instruct"
    ) async throws -> LanceUnderstanding {
        let loaded = try LanceLoader.loadModel(directory: directory)

        // Vision config from config.json's vision_config (HF `in_chans` → `in_channels`).
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard var root = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              var visionDict = root["vision_config"] as? [String: Any]
        else { throw LanceError.visionConfig("config.json has no vision_config") }
        if let inChans = visionDict.removeValue(forKey: "in_chans") {
            visionDict["in_channels"] = inChans
        }
        visionDict["model_type"] = visionDict["model_type"] ?? "qwen2_5_vl"
        let visionData = try JSONSerialization.data(withJSONObject: visionDict)
        let visionConfig = try JSONDecoder().decode(
            Qwen25VLConfiguration.VisionConfiguration.self, from: visionData)

        let vision = LanceVision.VisionModel(visionConfig)
        let vitWeights = try MLX.loadArrays(
            url: directory.appendingPathComponent("vit.safetensors"))
        let sanitized = vision.sanitize(weights: vitWeights)
        try vision.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.noUnusedKeys])
        eval(vision)

        let tokenizer = try await AutoTokenizer.from(pretrained: tokenizerSource)
        return try LanceUnderstanding(
            model: loaded.model, vision: vision, tokenizer: tokenizer,
            spatialMergeSize: visionConfig.spatialMergeSize)
    }

    /// VQA over one image. Returns the decoded answer text.
    public func generate(
        image: CIImage, question: String,
        instruction: String = LanceUnderstanding.defaultInstruction,
        maxNewTokens: Int = 256
    ) throws -> String {
        // 1. Lance-style template (image as 1-frame video).
        let text = "<|im_start|>system\n\(instruction)<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<|vision_start|><|image_pad|><|vision_end|>\(question)<|im_end|>\n"
            + "<|im_start|>assistant\n"

        // 2. Preprocess + ViT.
        let (patches, frame) = try LanceImageProcessing.preprocess(image: image)
        let visionDtype = vision.patchEmbed.proj.weight.dtype
        let imageFeatures = vision(patches.asType(visionDtype), frames: [frame])  // (N, D)

        // 3. Tokenize, expand the single image_pad to the per-patch count, then substitute
        //    image_pad → video_pad ids (Lance training convention).
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let padCount = frame.product / (spatialMergeSize * spatialMergeSize)
        if let idx = ids.firstIndex(of: imagePadId) {
            ids.replaceSubrange(idx...idx, with: Array(repeating: videoPadId, count: padCount))
        } else {
            throw LanceError.missingToken("<|image_pad|> not present after tokenization")
        }

        // 4. Embed + merge ViT features at the pad positions. Indexing follows the
        //    upstream `mergeInputIdsWithImageFeatures` pattern exactly (E6: an earlier
        //    `embeds[0, indices] = features` spelling compiled but did not condition
        //    generation — always assign (1, N, D) through [0..., indices, 0...] and use
        //    the returned array).
        let inputIds = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)  // (1, T)
        let textEmbeds = model.embedTokens(inputIds)
        let padPositions = ids.enumerated().filter { $0.element == videoPadId }.map(\.offset)
        guard padPositions.count == imageFeatures.dim(0) else {
            throw LanceError.imageProcessing(
                "pad count \(padPositions.count) != ViT tokens \(imageFeatures.dim(0))")
        }
        let embeds = try Self.mergeImageFeatures(
            textEmbeds: textEmbeds, imageFeatures: imageFeatures.asType(textEmbeds.dtype),
            padPositions: padPositions)

        if ProcessInfo.processInfo.environment["LANCE_DEBUG"] == "1" {
            // Branch-2 stage bisection (E6 run #7): compare against the Python reference
            // dumps (tools/dump_vit_reference.py --all). All 6 fixture cases share grid
            // 1×40×58, so the matching reference is selected by best pixel_values cosine,
            // then every captured stage is diffed — the first stage that falls off pins
            // the diverging op. window_index compares exactly; the rest by cosine.
            func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
                let x = a.asType(.float32).flattened()
                let y = b.asType(.float32).flattened()
                guard x.dim(0) == y.dim(0) else { return .nan }
                let d = (x * y).sum()
                let n = sqrt(x.square().sum()) * sqrt(y.square().sum())
                return (d / n).item(Float.self)
            }
            let refDir = (ProcessInfo.processInfo.environment["LANCE_VIT_REF"]
                .map { URL(fileURLWithPath: $0).deletingLastPathComponent() })
                ?? URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE")
            let refURLs = ((try? FileManager.default.contentsOfDirectory(
                at: refDir, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.lastPathComponent.hasPrefix("lance-vit-ref-case")
                    && $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            var best: (name: String, ref: [String: MLXArray], pixCos: Float)?
            for url in refURLs {
                guard let ref = try? MLX.loadArrays(url: url),
                      let refPixels = ref["pixel_values"] else { continue }
                let c = cosine(patches, refPixels)
                if best == nil || (!c.isNaN && c > best!.pixCos) {
                    best = (url.lastPathComponent, ref, c)
                }
            }
            if let best {
                print("[LANCE_DEBUG] ref-diff vs \(best.name) (best pixel match)")
                print("[LANCE_DEBUG]   pixel_values cosine=\(best.pixCos)")
                let stages = vision.debug.stages
                if let refIdx = best.ref["window_index"], let swiftIdx = stages["window_index"] {
                    let same = (swiftIdx.asType(.int32) .== refIdx.asType(.int32))
                        .asType(.float32).mean().item(Float.self)
                    print("[LANCE_DEBUG]   window_index exact-match=\(same) "
                        + (same == 1.0 ? "✓" : "✗ ORDERING DIVERGES"))
                }
                // Print every stage present on BOTH sides, sorted — robust to row evolution.
                let shared = Set(stages.keys).intersection(best.ref.keys)
                    .subtracting(["window_index"]).sorted()
                for stage in shared {
                    if let r = best.ref[stage], let s = stages[stage] {
                        print("[LANCE_DEBUG]   \(stage) cosine=\(cosine(s, r))")
                    }
                }
                if let refFeatures = best.ref["image_features"] {
                    print("[LANCE_DEBUG]   image_features cosine=\(cosine(imageFeatures, refFeatures))")
                }
                print("[LANCE_DEBUG]   slope reading: smooth monotone decline across blocks "
                    + "= bf16 kernel numerics (tag-with-caveat candidate); a STEP at one block "
                    + "or at a merger row = residual op bug at that stage.")
            }

            let featNorm = sqrt(imageFeatures.asType(.float32).square().sum()).item(Float.self)
            let idx = MLXArray(padPositions.map { Int32($0) })
            let delta = (embeds[0..., idx, 0...] - textEmbeds[0..., idx, 0...])
                .asType(.float32).square().sum()
            let textNorm = sqrt(textEmbeds[0..., idx, 0...].asType(.float32).square().sum())
            print("[LANCE_DEBUG] vit features: \(imageFeatures.dim(0))×\(imageFeatures.dim(1)) "
                + "L2=\(featNorm)")
            print("[LANCE_DEBUG] pad-position embeds: pre-merge L2=\(sqrt(textNorm).item(Float.self)) "
                + "merge delta L2=\(sqrt(delta).item(Float.self)) "
                + "(delta must be >0 and O(feature norm) — 0 means the merge is a no-op)")
        }

        // 5. 3D mRoPE position ids (port of _compute_position_ids, image path).
        let (positionIds, nextPosition) = Self.positionIds(
            ids: ids, frame: frame, mergeSize: spatialMergeSize,
            videoPadId: videoPadId)

        if ProcessInfo.processInfo.environment["LANCE_DEBUG"] == "1" {
            // Structural checks (E6 run #3 plan).
            print("[LANCE_DEBUG] T check: positionIds (3,1,\(positionIds.dim(2))) vs "
                + "embeds T=\(embeds.dim(1)) ids=\(ids.count) "
                + (positionIds.dim(2) == embeds.dim(1) ? "✓" : "✗ MISMATCH"))

            // Vision ablation: prefill with REAL vs ZEROED features (no cache), compare the
            // last-position logits. Identical → the answer position never attends vision
            // (attention/mask/position/cache bug). Different but text still wrong → vision is
            // attended but semantically inert (ViT correctness: sanitize/patchify/normalize).
            let zeroedEmbeds = try Self.mergeImageFeatures(
                textEmbeds: textEmbeds,
                imageFeatures: MLXArray.zeros(
                    [imageFeatures.dim(0), imageFeatures.dim(1)], dtype: textEmbeds.dtype),
                padPositions: padPositions)
            func lastLogits(_ e: MLXArray) -> MLXArray {
                let h = model(
                    inputEmbeddings: e, positionIds: positionIds, positionGroup: nil,
                    mask: .causal, caches: nil)
                return model.logits(h[0..., -1, 0...]).asType(.float32)
            }
            let realL = lastLogits(embeds)
            let zeroL = lastLogits(zeroedEmbeds)
            let maxDiff = abs(realL - zeroL).max().item(Float.self)
            let argReal = argMax(realL, axis: -1).item(Int.self)
            let argZero = argMax(zeroL, axis: -1).item(Int.self)
            print("[LANCE_DEBUG] ablation (real vs zeroed features): "
                + "max|Δlogit|=\(maxDiff) argmax real=\(argReal) zero=\(argZero) "
                + (maxDiff < 1e-3
                    ? "→ IDENTICAL: vision never reaches the query (attention/mask/pos/cache)"
                    : "→ vision IS attended; if text still wrong, suspect ViT semantics"))
        }

        // 6. Greedy decode with KV cache; both EOS ids stop.
        let caches = (0..<model.config.numHiddenLayers).map { _ in LanceKVCache() }
        var hidden = model(
            inputEmbeddings: embeds, positionIds: positionIds, positionGroup: nil,
            mask: .causal, caches: caches)
        var output: [Int] = []
        var position = nextPosition
        for _ in 0..<maxNewTokens {
            let logits = model.logits(hidden[0..., -1, 0...])
            let next = argMax(logits, axis: -1).item(Int.self)
            if next == LanceTokens.imEnd || next == LanceTokens.endOfText { break }
            output.append(next)

            let nextEmbed = model.embedTokens(
                MLXArray([Int32(next)]).expandedDimensions(axis: 0))
            // Single step: same scalar position on all three axes.
            let stepPos = MLXArray([Int32(position)]).reshaped(1, 1, 1)
            let stepIds = broadcast(stepPos, to: [3, 1, 1])
            hidden = model(
                inputEmbeddings: nextEmbed, positionIds: stepIds, positionGroup: nil,
                mask: .none, caches: caches)
            position += 1
        }
        return tokenizer.decode(tokens: output)
    }

    /// Slot ViT features into the text embeddings at the pad positions.
    ///
    /// **E6 (run #2 evidence):** subscript-set scatter (`result[0..., indices, 0...] = features`,
    /// and the `[0, indices]` spelling before it) is a **silent no-op** in mlx-swift here —
    /// merge delta 0.0 on all six oracle cases while feature norms were healthy (900–2500).
    /// So: no setters. The pad block is contiguous by construction (one `replaceSubrange`
    /// inserts it), so the merge is pure slice + concatenate along the sequence axis.
    static func mergeImageFeatures(
        textEmbeds: MLXArray, imageFeatures: MLXArray, padPositions: [Int]
    ) throws -> MLXArray {
        guard let start = padPositions.first, let last = padPositions.last,
              padPositions.count == last - start + 1
        else {
            throw LanceError.imageProcessing(
                "pad positions not a single contiguous block (\(padPositions.count) positions)")
        }
        let features = imageFeatures.ndim == 2
            ? imageFeatures[.newAxis, 0..., 0...]   // (1, N, D)
            : imageFeatures
        var parts: [MLXArray] = []
        if start > 0 { parts.append(textEmbeds[0..., ..<start, 0...]) }
        parts.append(features)
        let end = last + 1
        if end < textEmbeds.dim(1) { parts.append(textEmbeds[0..., end..., 0...]) }
        return concatenated(parts, axis: 1)
    }

    /// 3D position ids for a single-image prompt: text positions advance on all axes;
    /// vision tokens get the (t, h, w) grid (h/w divided by the merge size), anchored at
    /// the text position where the vision block starts. Returns (3, 1, T) + next position.
    static func positionIds(
        ids: [Int], frame: THW, mergeSize: Int, videoPadId: Int
    ) -> (MLXArray, Int) {
        let llmGridH = frame.h / mergeSize
        let llmGridW = frame.w / mergeSize

        var t = [Int32](); var h = [Int32](); var w = [Int32]()
        t.reserveCapacity(ids.count); h.reserveCapacity(ids.count); w.reserveCapacity(ids.count)

        var textPos: Int32 = 0
        var i = 0
        while i < ids.count {
            if ids[i] == videoPadId {
                // The whole contiguous vision block, laid out t-major over the grid.
                let anchor = textPos
                for ti in 0..<frame.t {
                    for hi in 0..<llmGridH {
                        for wi in 0..<llmGridW {
                            t.append(anchor + Int32(ti))
                            h.append(anchor + Int32(hi))
                            w.append(anchor + Int32(wi))
                        }
                    }
                }
                i += frame.t * llmGridH * llmGridW
                // Text resumes after the largest grid coordinate.
                textPos = anchor + Int32(max(frame.t, max(llmGridH, llmGridW)))
            } else {
                t.append(textPos); h.append(textPos); w.append(textPos)
                textPos += 1
                i += 1
            }
        }
        let stacked = stacked(
            [MLXArray(t), MLXArray(h), MLXArray(w)], axis: 0
        ).reshaped(3, 1, ids.count)
        return (stacked, Int(textPos))
    }
}
