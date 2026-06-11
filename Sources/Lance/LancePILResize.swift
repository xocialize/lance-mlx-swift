import Foundation

/// PIL-exact bicubic resize on interleaved RGB8 — a faithful port of Pillow's `Resample.c`
/// 8-bits-per-channel path (the resampler the HF Qwen2.5-VL processor uses before
/// rescale/normalize).
///
/// **Why this exists (E6 root cause, L1 run #12):** CoreImage bicubic ≠ PIL bicubic. The
/// ~6e-4 high-frequency pixel deviation it caused was amplified by the patch-embed
/// projection (0.9994 → 0.9938) and block-17's massive-activation MLP into image_features
/// 0.91–0.98 — enough to flip a boundary VQA answer. VLM preprocessing must match the
/// reference resampler exactly; verify, don't threshold.
///
/// Faithfulness notes (all from Pillow's Resample.c):
/// - bicubic kernel a = −0.5, support 2.0
/// - `filterscale = max(scale, 1)` widens the kernel when downscaling (PIL always
///   antialiases); weights normalized to sum 1 per output pixel
/// - 8bpc fixed point: PRECISION_BITS = 32 − 8 − 2 = 22; coefficients rounded with ±0.5;
///   accumulator seeded with `1 << (PRECISION_BITS − 1)`; result `>> PRECISION_BITS`,
///   clamped to 0…255
/// - horizontal pass first, then vertical
public enum LancePILResize {
    static let precisionBits = 32 - 8 - 2  // 22

    static func bicubic(_ xIn: Double) -> Double {
        let a = -0.5
        let x = abs(xIn)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Per-output-pixel integer coefficients + source bounds (PIL `precompute_coeffs` +
    /// `normalize_coeffs_8bpc`).
    static func coefficients(inSize: Int, outSize: Int)
        -> (bounds: [(min: Int, count: Int)], coeffs: [[Int32]])
    {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let one = Double(1 << precisionBits)

        var bounds: [(Int, Int)] = []
        var coeffs: [[Int32]] = []
        bounds.reserveCapacity(outSize)
        coeffs.reserveCapacity(outSize)

        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)   // C truncation; clamped next
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            let count = xmax - xmin

            var w = [Double](repeating: 0, count: count)
            var total = 0.0
            for x in 0..<count {
                let v = bicubic((Double(x + xmin) - center + 0.5) / filterscale)
                w[x] = v
                total += v
            }
            var k = [Int32](repeating: 0, count: count)
            for x in 0..<count {
                let normalized = total != 0 ? w[x] / total : w[x]
                // PIL: (int)(k < 0 ? k*one - 0.5 : k*one + 0.5) — round half away from zero.
                let scaled = normalized * one
                k[x] = Int32(scaled < 0 ? scaled - 0.5 : scaled + 0.5)
            }
            bounds.append((xmin, count))
            coeffs.append(k)
        }
        return (bounds, coeffs)
    }

    @inline(__always)
    static func clip8(_ v: Int32) -> UInt8 {
        let shifted = v >> Int32(precisionBits)
        return UInt8(min(max(shifted, 0), 255))
    }

    /// Resize interleaved RGB8 (row-major, 3 bytes/pixel) to (outWidth, outHeight).
    public static func resize(
        rgb: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int
    ) -> [UInt8] {
        let half = Int32(1 << (precisionBits - 1))

        // Horizontal pass: (height, width) → (height, outWidth)
        let (hBounds, hCoeffs) = coefficients(inSize: width, outSize: outWidth)
        var temp = [UInt8](repeating: 0, count: height * outWidth * 3)
        rgb.withUnsafeBufferPointer { src in
            temp.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let rowIn = y * width * 3
                    let rowOut = y * outWidth * 3
                    for xx in 0..<outWidth {
                        let (xmin, count) = hBounds[xx]
                        let k = hCoeffs[xx]
                        var s0 = half, s1 = half, s2 = half
                        for x in 0..<count {
                            let p = rowIn + (xmin + x) * 3
                            let w = k[x]
                            s0 += Int32(src[p]) * w
                            s1 += Int32(src[p + 1]) * w
                            s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + xx * 3
                        dst[o] = clip8(s0)
                        dst[o + 1] = clip8(s1)
                        dst[o + 2] = clip8(s2)
                    }
                }
            }
        }

        // Vertical pass: (height, outWidth) → (outHeight, outWidth)
        let (vBounds, vCoeffs) = coefficients(inSize: height, outSize: outHeight)
        var out = [UInt8](repeating: 0, count: outHeight * outWidth * 3)
        temp.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for yy in 0..<outHeight {
                    let (ymin, count) = vBounds[yy]
                    let k = vCoeffs[yy]
                    let rowOut = yy * outWidth * 3
                    for xx in 0..<outWidth {
                        let col = xx * 3
                        var s0 = half, s1 = half, s2 = half
                        for y in 0..<count {
                            let p = (ymin + y) * outWidth * 3 + col
                            let w = k[y]
                            s0 += Int32(src[p]) * w
                            s1 += Int32(src[p + 1]) * w
                            s2 += Int32(src[p + 2]) * w
                        }
                        let o = rowOut + col
                        dst[o] = clip8(s0)
                        dst[o + 1] = clip8(s1)
                        dst[o + 2] = clip8(s2)
                    }
                }
            }
        }
        return out
    }
}
