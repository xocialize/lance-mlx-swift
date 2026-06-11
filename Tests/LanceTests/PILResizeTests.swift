import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import Lance

// E6 resampler gate — pure CPU, no MLX, no Metal: the Swift PIL-exact bicubic must match
// Pillow byte-for-byte (±1 LSB tolerance for decode/color-management differences) on the
// six L1 fixtures BEFORE any inference. References from tools/dump_pil_resize.py.

private let refDir = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/lance-pil-resize")
private let fixturesDir = URL(
    fileURLWithPath: "/Volumes/DEV_ARCHIVE/lance-mlx/tests/fixtures/images")

private func decodeRGB8(_ url: URL) throws -> (rgb: [UInt8], width: Int, height: Int) {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw LanceError.imageProcessing("cannot decode \(url.lastPathComponent)") }
    let width = image.width
    let height = image.height
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let context = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0..<(width * height) {
        rgb[i * 3] = rgba[i * 4]
        rgb[i * 3 + 1] = rgba[i * 4 + 1]
        rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return (rgb, width, height)
}

@Suite(.enabled(if: FileManager.default.fileExists(atPath: refDir.path)))
struct PILResizeTests {
    @Test(arguments: ["01", "02", "03", "04", "05", "06"])
    func matchesPILOnFixture(_ caseID: String) throws {
        let meta = try JSONDecoder().decode(
            [String: Int].self,
            from: Data(contentsOf: refDir.appendingPathComponent("case\(caseID).json"))
                .jsonDimsOnly())
        let dstW = meta["dst_w"]!
        let dstH = meta["dst_h"]!
        let reference = [UInt8](
            try Data(contentsOf: refDir.appendingPathComponent("case\(caseID).bin")))

        let (rgb, width, height) = try decodeRGB8(
            fixturesDir.appendingPathComponent("image-understanding-case-\(caseID).png"))
        let resized = LancePILResize.resize(
            rgb: rgb, width: width, height: height, outWidth: dstW, outHeight: dstH)

        #expect(resized.count == reference.count)
        var maxDiff = 0
        var diffCount = 0
        for i in 0..<min(resized.count, reference.count) {
            let d = abs(Int(resized[i]) - Int(reference[i]))
            if d > 0 { diffCount += 1 }
            maxDiff = max(maxDiff, d)
        }
        let diffShare = Double(diffCount) / Double(reference.count)
        // ±1 LSB on a small share of pixels = decode/rounding noise; >1 = algorithm divergence.
        #expect(maxDiff <= 1, "case \(caseID): max|Δ|=\(maxDiff) (diff share \(diffShare))")
    }
}

extension Data {
    /// The dump's json mixes strings and ints; keep only int fields for simple decoding.
    fileprivate func jsonDimsOnly() throws -> Data {
        let obj = try JSONSerialization.jsonObject(with: self) as! [String: Any]
        let ints = obj.compactMapValues { $0 as? Int }
        return try JSONSerialization.data(withJSONObject: ints)
    }
}
