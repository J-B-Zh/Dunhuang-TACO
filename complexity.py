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


def residual_merge_macs(resolution: int, inputs: int, outputs: int) -> int:
    """MACs of the residual convolutional Patch Merging block."""
    return (
        conv_macs(resolution, resolution, inputs, outputs, 3)
        + conv_macs(resolution, resolution, outputs, outputs, 3)
        + conv_macs(resolution, resolution, inputs, outputs, 1)
    )


def residual_expand_macs(resolution: int, inputs: int, outputs: int) -> int:
    """MACs of the residual convolutional Patch Expansion block.

    Transposed-convolution MACs are counted at their input resolution.
    """
    return (
        conv_macs(resolution, resolution, inputs, outputs, 3)
        + conv_macs(resolution, resolution, outputs, outputs, 3)
        + conv_macs(resolution, resolution, inputs, outputs, 3)
    )


def swin_block_macs(resolution: int, dim: int, depth: int, window_size: int) -> int:
    """MACs for shifted-window attention plus its 4x MLP."""
    tokens = resolution * resolution
    window = min(resolution, window_size)
    block = 12 * tokens * dim * dim + 2 * tokens * window * window * dim
    return depth * block


def convolution_stage_macs(resolution: int, dim: int, depth: int) -> int:
    """MACs for the U-Net-style convolution blocks inside one stage."""
    return depth * conv_macs(resolution, resolution, dim, dim, 3)


def mamba_block_macs(resolution: int, dim: int, depth: int) -> int:
    """MACs for the four-directional Mamba blocks inside one stage."""
    tokens = resolution * resolution
    inner = dim * 2
    state_dim = 16
    dt_rank = (dim + 15) // 16
    one_scan = (
        tokens * dim * (2 * inner)
        + tokens * inner * 3
        + tokens * inner * (dt_rank + 2 * state_dim)
        + tokens * dt_rank * inner
        + 9 * tokens * inner * state_dim
        + tokens * inner
    )
    block = 4 * one_scan + tokens * inner * dim
    return depth * block


def stage_macs(
    name: str, resolution: int, dim: int, depth: int, window_size: int
) -> int:
    if name == "swin":
        return swin_block_macs(resolution, dim, depth, window_size)
    if name == "unet":
        return convolution_stage_macs(resolution, dim, depth)
    if name == "mamba":
        return mamba_block_macs(resolution, dim, depth)
    raise ValueError(f"Unknown backbone: {name}")


def hierarchical_backbone_macs(
    name: str, patch_size: int, dim: int, depth: int
) -> tuple[int, int]:
    """Common hierarchy in which only encoder/decoder stage blocks change."""
    channels = (dim, dim * 2, dim * 4, dim * 8)
    resolutions = (
        patch_size // 4,
        patch_size // 8,
        patch_size // 16,
        patch_size // 32,
    )
    encoder = conv_macs(resolutions[0], resolutions[0], 3, dim, 4)
    for index, (resolution, channels_in) in enumerate(zip(resolutions, channels)):
        encoder += stage_macs(
            name, resolution, channels_in, depth, 16 if index == 0 else 8
        )
        if index < 3:
            encoder += residual_merge_macs(
                resolution, channels_in, channels[index + 1]
            )

    decoder = 0
    decoder_channels = (dim * 16, dim * 8, dim * 4, dim * 2)
    decoder_resolutions = tuple(reversed(resolutions))
    for resolution, channels_in in zip(decoder_resolutions, decoder_channels):
        decoder += stage_macs(name, resolution, channels_in, depth, 8)
        decoder += residual_expand_macs(
            resolution, channels_in, channels_in // 4
        )
    decoder += residual_expand_macs(patch_size // 2, dim // 2, dim // 8)
    decoder += conv_macs(patch_size, patch_size, dim // 8, 3, 3)
    return encoder, decoder


def backbone_macs(
    name: str, patch_size: int, dim: int, depth: int, window_size: int = 8
) -> tuple[int, int]:
    """Return encoder and decoder MACs for one 256-pixel patch."""
    del window_size
    return hierarchical_backbone_macs(name, patch_size, dim, depth)


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


def adapter_macs(patch_count: int, input_dim: int, graph_dim: int) -> int:
    """Two encoder-to-graph passes and one graph-to-decoder pass."""
    if input_dim == graph_dim:
        return 0
    return 3 * patch_count * input_dim * graph_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=256, choices=(256,))
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument(
        "--depth", type=int, default=2,
        help="blocks per stage for all controlled backbone variants",
    )
    parser.add_argument("--heads", type=int, default=None)
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
            + graph_macs(patch_count, model.graph_dim)
            + adapter_macs(patch_count, model.feature_dim, model.graph_dim)
        )
        print(
            f"{name} | {backbone_params / 1e6:.3f} | "
            f"{backbone_total / 1e9:.3f} | {trainable_parameters(model) / 1e6:.3f} | "
            f"{generator_total / 1e9:.3f}"
        )


if __name__ == "__main__":
    main()
