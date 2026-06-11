// swift-tools-version: 6.0
// lance-mlx-swift — Swift/MLX port of Lance (ByteDance's unified multimodal model:
// dual-tower MoT over Qwen2.5-VL-3B, image/video understanding + generation + editing).
// Ported from our production-validated Python port (xocialize/lance-mlx); consumes the
// published mlx-community checkpoints (Lance-3B-bf16 et al.) exactly as-is.
// L1 scope: x2t_image understanding. See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "Lance",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "Lance", targets: ["Lance"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Public Qwen2.5-VL utilities (PatchEmbed, patchify/targetSize, processor, KVCache).
        // fileprivate vision/language blocks are copy-adapted into Sources/Lance/Adapted/
        // with attribution (MIT) — see NOTICE.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "Lance",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Lance"
        ),
        .testTarget(
            name: "LanceTests",
            dependencies: ["Lance"],
            path: "Tests/LanceTests"
        ),
    ]
)
