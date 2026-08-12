"""Report parameter counts and MACs for each Dunhuang-TACO backbone.

MACs are estimated analytically for one complete generator forward pass with a
degraded image, mask, and retrieved reference image. DINO-v2 retrieval is
excluded because it is shared by all variants. Many papers and profiling tools
report one multiply-accumulate as one FLOP; under the two-operation convention,
multiply the reported MAC values by two.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from dunhuang_taco.model import DunhuangTACO


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def conv_macs(height: int, width: int, inputs: int, outputs: int, kernel: int) -> int:
    return height * width * inputs * outputs * kernel * kernel


def backbone_macs(
    name: str, patch_size: int, dim: int, depth: int, window_size: int = 8
) -> tuple[int, int]:
    """Return encoder and decoder MACs for one 256-pixel patch."""
    feature_size = patch_size // 4
    tokens = feature_size * feature_size
    embed = conv_macs(feature_size, feature_size, 3, dim, 4)
    output = conv_macs(feature_size, feature_size, dim, 48, 3)

    if name == "swin":
        # QKV/output projections, two MLP layers, and window attention products.
        block = 12 * tokens * dim * dim + 2 * tokens * window_size * window_size * dim
        return embed + depth * block, depth * block + output

    if name == "unet":
        base = max(16, dim // 2)
        half = patch_size // 2
        quarter = patch_size // 4
        encoder = (
            conv_macs(patch_size, patch_size, 3, base, 3)
            + conv_macs(patch_size, patch_size, base, base, 3)
            + conv_macs(half, half, base, dim, 3)
            + conv_macs(half, half, dim, dim, 3)
            + 2 * conv_macs(quarter, quarter, dim, dim, 3)
        )
        decoder = (
            conv_macs(half, half, dim, dim, 2)
            + conv_macs(half, half, dim * 2, dim, 3)
            + conv_macs(half, half, dim, dim, 3)
            + conv_macs(patch_size, patch_size, dim, base, 2)
            + conv_macs(patch_size, patch_size, base * 2, base, 3)
            + conv_macs(patch_size, patch_size, base, base, 3)
            + conv_macs(patch_size, patch_size, base, 3, 1)
        )
        return encoder, decoder

    if name == "mamba":
        inner = dim * 2
        state_dim = 16
        dt_rank = (dim + 15) // 16
        # Four directional selective scans. The 9*N*I*S term follows the
        # standard selective-scan MAC estimate and includes state updates.
        one_scan = (
            tokens * dim * (2 * inner)
            + tokens * inner * 3
            + tokens * inner * (dt_rank + 2 * state_dim)
            + tokens * dt_rank * inner
            + 9 * tokens * inner * state_dim
            + tokens * inner
        )
        block = 4 * one_scan + tokens * inner * dim
        return embed + depth * block, depth * block + output

    raise ValueError(f"Unknown backbone: {name}")


def graph_macs(patch_count: int, dim: int, topk_ratio: float = 0.5) -> int:
    """MACs of the shared intra- and cross-mural sparse graph branches."""
    intact = max(1, patch_count // 2)
    intra_neighbors = max(1, int(intact * topk_ratio + 0.999999))
    cross_neighbors = max(1, int(patch_count * topk_ratio + 0.999999))

    def branch(keys: int, neighbors: int) -> int:
        similarity = patch_count * keys * dim
        affine = patch_count * neighbors * dim * (2 * dim)
        output = patch_count * dim * dim
        return similarity + affine + output

    return branch(intact, intra_neighbors) + branch(patch_count, cross_neighbors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256, choices=(256,))
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution % args.patch_size:
        raise ValueError("resolution must be divisible by patch size")
    device = torch.device(args.device)
    patch_count = (args.resolution // args.patch_size) ** 2

    print(
        "Backbone | Backbone Params (M) | Backbone MACs (G) | "
        "Generator Params (M) | Generator MACs (G)"
    )
    print("---|---:|---:|---:|---:")
    for name in ("unet", "mamba", "swin"):
        model = DunhuangTACO(
            backbone=name,
            patch_batch_size=1,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
        ).to(device).eval()
        backbone_params = trainable_parameters(model.encoder) + trainable_parameters(model.decoder)
        encoder_macs, decoder_macs = backbone_macs(
            name, args.patch_size, args.dim, args.depth
        )
        backbone_total = patch_count * (encoder_macs + decoder_macs)
        generator_total = (
            patch_count * (2 * encoder_macs + decoder_macs)
            + graph_macs(patch_count, args.dim)
        )
        print(
            f"{name} | {backbone_params / 1e6:.3f} | "
            f"{backbone_total / 1e9:.3f} | {trainable_parameters(model) / 1e6:.3f} | "
            f"{generator_total / 1e9:.3f}"
        )


if __name__ == "__main__":
    main()
