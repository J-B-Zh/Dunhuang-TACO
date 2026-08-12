"""Dunhuang-TACO generator and discriminator.

The implementation mirrors Sections 3.2--3.5 of the manuscript: a four-stage
hierarchical backbone adapted from ``code/xiufu-dan``, parallel SCA/R-TAP
sparse graphs, and mask-guided convex fusion. All variants share residual
convolutional Patch Merging/Expansion and replace only the blocks inside each
stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .hierarchical_swin import (
    HierarchicalMambaDecoder,
    HierarchicalMambaEncoder,
    HierarchicalSwinDecoder,
    HierarchicalSwinEncoder,
    HierarchicalUNetDecoder,
    HierarchicalUNetEncoder,
)


BACKBONES: dict[str, tuple[type[nn.Module], type[nn.Module]]] = {
    "swin": (HierarchicalSwinEncoder, HierarchicalSwinDecoder),
    "unet": (HierarchicalUNetEncoder, HierarchicalUNetDecoder),
    "mamba": (HierarchicalMambaEncoder, HierarchicalMambaDecoder),
}


class SparseGraphAggregation(nn.Module):
    """Top-K relation-aware modulation shared by SCA and R-TAP.

    Queries and keys are patch tokens with shape ``[B,N,D]``. For every query,
    the module selects ``ceil(topk_ratio * N_key)`` neighbors using dot-product
    similarity, constructs the paper's directed edge features, and
    applies learned affine modulation to the normalized query.
    """

    def __init__(self, dim: int, topk_ratio: float = 0.5):
        super().__init__()
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        self.topk_ratio = topk_ratio
        self.norm = nn.LayerNorm(dim)
        self.affine = nn.Sequential(nn.GELU(), nn.Linear(dim, dim * 2))
        self.output = nn.Linear(dim, dim)

    def forward(
        self, queries: Tensor, keys: Tensor, key_valid: Tensor | None = None
    ) -> Tensor:
        if queries.ndim != 3 or keys.ndim != 3:
            raise ValueError("queries and keys must have shape [B, N, D]")
        if queries.shape[0] != keys.shape[0] or queries.shape[2] != keys.shape[2]:
            raise ValueError("query/key batch and channel dimensions must match")

        if key_valid is not None:
            if key_valid.shape != keys.shape[:2]:
                raise ValueError("key_valid must have shape [B, N_key]")
        else:
            key_valid = torch.ones(keys.shape[:2], dtype=torch.bool, device=keys.device)

        outputs = []
        for query, key, valid in zip(queries, keys, key_valid):
            valid_keys = key[valid]
            if len(valid_keys) == 0:
                outputs.append(query)
                continue
            k = max(1, math.ceil(len(valid_keys) * self.topk_ratio))
            similarity = query @ valid_keys.transpose(0, 1)
            scores, indices = similarity.topk(k, dim=-1)
            neighbors = valid_keys[indices]
            probabilities = scores.softmax(dim=-1).unsqueeze(-1)
            query_edges = query.unsqueeze(1).expand(-1, k, -1)
            edges = (1.0 - probabilities) * query_edges + probabilities * neighbors
            alpha, beta = self.affine(edges).chunk(2, dim=-1)
            messages = alpha * self.norm(query).unsqueeze(1) + beta
            outputs.append(query + F.gelu(self.output(messages.sum(dim=1))))
        return torch.stack(outputs)


class MaskGuidedFusion(nn.Module):
    """Convexly combine cross- and intra-mural features using local damage."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(1, 1)

    def forward(self, cross: Tensor, intra: Tensor, damage_ratio: Tensor) -> tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.gate(damage_ratio.unsqueeze(-1)))
        return gate * cross + (1.0 - gate) * intra, gate


@dataclass
class ModelOutput:
    prediction: Tensor
    completed: Tensor
    gate: Tensor
    patch_damage: Tensor


class DunhuangTACO(nn.Module):
    """Theme-aware coherent mural inpainting generator."""

    def __init__(
        self,
        patch_size: int = 256,
        patch_batch_size: int = 16,
        dim: int = 96,
        depth: int | None = None,
        heads: int | tuple[int, ...] | None = None,
        topk_ratio: float = 0.5,
        backbone: str = "swin",
    ):
        super().__init__()
        if patch_size != 256:
            raise ValueError("The manuscript configuration uses patch_size=256")
        self.patch_size = patch_size
        self.patch_batch_size = patch_batch_size
        backbone = backbone.lower()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {backbone!r}")
        encoder_type, decoder_type = BACKBONES[backbone]
        self.backbone = backbone
        if depth is None:
            depth = 2
        self.encoder = encoder_type(dim, depth, heads)
        self.decoder = decoder_type(dim, depth, heads)
        feature_dim = getattr(self.encoder, "output_dim", dim)
        self.feature_dim = feature_dim
        # SCA/R-TAP always operate at the 768-D xiufu-dan bottleneck. Lightweight
        # adapters make the U-Net/Mamba replacements comparable without changing
        # their native encoder/decoder widths.
        self.graph_dim = dim * 8
        self.encoder_to_graph = (
            nn.Identity() if feature_dim == self.graph_dim else nn.Linear(feature_dim, self.graph_dim)
        )
        self.graph_to_decoder = (
            nn.Identity() if feature_dim == self.graph_dim else nn.Linear(self.graph_dim, feature_dim)
        )
        self.graph = SparseGraphAggregation(self.graph_dim, topk_ratio)
        self.fusion = MaskGuidedFusion()

    def _split(self, image: Tensor) -> tuple[Tensor, int, int]:
        b, c, h, w = image.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("image height and width must be multiples of 256")
        gh, gw = h // self.patch_size, w // self.patch_size
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b * gh * gw, c, self.patch_size, self.patch_size)
        return patches, gh, gw

    @staticmethod
    def _join(patches: Tensor, b: int, gh: int, gw: int) -> Tensor:
        _, c, ph, pw = patches.shape
        return patches.view(b, gh, gw, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, gh * ph, gw * pw)

    def _chunked_encode(
        self, values: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        maps = []
        skip_groups: list[list[Tensor]] = []
        for start in range(0, len(values), self.patch_batch_size):
            encoded = self.encoder(values[start : start + self.patch_batch_size])
            if isinstance(encoded, tuple):
                feature_map, skips = encoded
                maps.append(feature_map)
                if not skip_groups:
                    skip_groups = [[] for _ in skips]
                for group, skip in zip(skip_groups, skips):
                    group.append(skip)
            else:
                maps.append(encoded)
        return torch.cat(maps, dim=0), (
            tuple(torch.cat(group, dim=0) for group in skip_groups)
            if skip_groups
            else None
        )

    def _chunked_decode(
        self, values: Tensor, skips: tuple[Tensor, ...] | None = None
    ) -> Tensor:
        outputs = []
        for start in range(0, len(values), self.patch_batch_size):
            chunk = values[start : start + self.patch_batch_size]
            if skips is None:
                outputs.append(self.decoder(chunk))
            else:
                skip_chunk = tuple(
                    skip[start : start + self.patch_batch_size] for skip in skips
                )
                outputs.append(self.decoder(chunk, skip_chunk))
        return torch.cat(outputs, dim=0)

    def _encode_tokens(
        self, image: Tensor
    ) -> tuple[Tensor, tuple[Tensor, ...] | None, Tensor, int, int]:
        patches, gh, gw = self._split(image)
        maps, skips = self._chunked_encode(patches)
        tokens = maps.mean(dim=(-2, -1)).view(image.shape[0], gh * gw, -1)
        tokens = self.encoder_to_graph(tokens)
        return maps, skips, tokens, gh, gw

    def forward(self, degraded: Tensor, mask: Tensor, reference: Tensor) -> ModelOutput:
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError("degraded must have shape [B,3,H,W]")
        if mask.shape != degraded.shape[:1] + (1,) + degraded.shape[2:]:
            raise ValueError("mask must have shape [B,1,H,W]")
        if reference.shape != degraded.shape:
            reference = F.interpolate(reference, degraded.shape[-2:], mode="bilinear", align_corners=False)

        masked = degraded * (1.0 - mask)
        encoded, skips, tokens, gh, gw = self._encode_tokens(masked)
        _, _, ref_tokens, _, _ = self._encode_tokens(reference)

        mask_patches, _, _ = self._split(mask)
        damage = mask_patches.mean(dim=(-3, -2, -1)).view(degraded.shape[0], gh * gw)
        degraded_nodes = damage > 0
        intact_nodes = damage == 0

        # Both branches consume the same degraded tokens and run independently.
        intra = self.graph(tokens, tokens, intact_nodes)
        cross = self.graph(tokens, ref_tokens)
        fused, gate = self.fusion(cross, intra, damage)
        tokens = torch.where(degraded_nodes.unsqueeze(-1), fused, tokens)

        decoder_tokens = self.graph_to_decoder(tokens)
        delta = (
            decoder_tokens.reshape(-1, decoder_tokens.shape[-1])
            - encoded.mean(dim=(-2, -1))
        ).unsqueeze(-1).unsqueeze(-1)
        decoded = self._chunked_decode(encoded + delta, skips)
        prediction = self._join(decoded, degraded.shape[0], gh, gw)
        completed = degraded * (1.0 - mask) + prediction * mask
        return ModelOutput(prediction, completed, gate.squeeze(-1), damage)


class PatchDiscriminator(nn.Module):
    """Small fully convolutional discriminator for the LSGAN objective."""

    def __init__(self, base: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for multiplier in (2, 4, 8):
            layers.extend(
                [
                    nn.Conv2d(base * multiplier // 2, base * multiplier, 4, 2, 1),
                    nn.InstanceNorm2d(base * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.extend([nn.Conv2d(base * 8, 1, 3, 1, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.net = nn.Sequential(*layers)

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)
