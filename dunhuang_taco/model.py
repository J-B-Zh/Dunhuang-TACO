"""Dunhuang-TACO generator and discriminator.

The implementation mirrors Sections 3.2--3.5 of the manuscript: a shared
chunk-wise Swin patch backbone, parallel SCA/R-TAP sparse graphs, and
mask-guided convex fusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _windows(x: Tensor, size: int) -> Tensor:
    """[B,H,W,C] -> [B*nW,size*size,C]."""
    b, h, w, c = x.shape
    x = x.view(b, h // size, size, w // size, size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, c)


def _unwindows(x: Tensor, size: int, h: int, w: int) -> Tensor:
    """[B*nW,size*size,C] -> [B,H,W,C]."""
    b = x.shape[0] // ((h // size) * (w // size))
    x = x.view(b, h // size, w // size, size, size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class SwinBlock(nn.Module):
    """Compact shifted-window Transformer block for one 256-pixel patch."""

    def __init__(self, dim: int, heads: int, window_size: int = 8, shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.shift = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1)
        if self.shift:
            y = torch.roll(y, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = _windows(y, self.window_size)
        z = self.norm1(win)
        win = win + self.attn(z, z, z, need_weights=False)[0]
        win = win + self.mlp(self.norm2(win))
        y = _unwindows(win, self.window_size, h, w)
        if self.shift:
            y = torch.roll(y, shifts=(self.shift, self.shift), dims=(1, 2))
        return y.permute(0, 3, 1, 2).contiguous()


class SwinPatchEncoder(nn.Module):
    """Shared L-layer encoder applied to each 256x256 mural patch."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList(
            SwinBlock(dim, heads, shift=bool(i % 2)) for i in range(depth)
        )

    def forward(self, patches: Tensor) -> Tensor:
        x = self.embed(patches)
        for block in self.blocks:
            x = block(x)
        return x


class SwinPatchDecoder(nn.Module):
    """Shared L-layer decoder symmetric to :class:`SwinPatchEncoder`."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinBlock(dim, heads, shift=bool(i % 2)) for i in range(depth)
        )
        self.output = nn.Sequential(nn.Conv2d(dim, 3 * 16, 3, padding=1), nn.PixelShuffle(4))

    def forward(self, features: Tensor) -> Tensor:
        x = features
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.output(x))


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
        depth: int = 4,
        heads: int = 4,
        topk_ratio: float = 0.5,
    ):
        super().__init__()
        if patch_size != 256:
            raise ValueError("The manuscript configuration uses patch_size=256")
        self.patch_size = patch_size
        self.patch_batch_size = patch_batch_size
        self.encoder = SwinPatchEncoder(dim, depth, heads)
        self.decoder = SwinPatchDecoder(dim, depth, heads)
        self.graph = SparseGraphAggregation(dim, topk_ratio)
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

    def _chunked(self, module: nn.Module, values: Tensor) -> Tensor:
        return torch.cat(
            [module(values[i : i + self.patch_batch_size]) for i in range(0, len(values), self.patch_batch_size)],
            dim=0,
        )

    def _encode_tokens(self, image: Tensor) -> tuple[Tensor, Tensor, int, int]:
        patches, gh, gw = self._split(image)
        maps = self._chunked(self.encoder, patches)
        tokens = maps.mean(dim=(-2, -1)).view(image.shape[0], gh * gw, -1)
        return maps, tokens, gh, gw

    def forward(self, degraded: Tensor, mask: Tensor, reference: Tensor) -> ModelOutput:
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError("degraded must have shape [B,3,H,W]")
        if mask.shape != degraded.shape[:1] + (1,) + degraded.shape[2:]:
            raise ValueError("mask must have shape [B,1,H,W]")
        if reference.shape != degraded.shape:
            reference = F.interpolate(reference, degraded.shape[-2:], mode="bilinear", align_corners=False)

        masked = degraded * (1.0 - mask)
        encoded, tokens, gh, gw = self._encode_tokens(masked)
        _, ref_tokens, _, _ = self._encode_tokens(reference)

        mask_patches, _, _ = self._split(mask)
        damage = mask_patches.mean(dim=(-3, -2, -1)).view(degraded.shape[0], gh * gw)
        degraded_nodes = damage > 0
        intact_nodes = damage == 0

        # Both branches consume the same degraded tokens and run independently.
        intra = self.graph(tokens, tokens, intact_nodes)
        cross = self.graph(tokens, ref_tokens)
        fused, gate = self.fusion(cross, intra, damage)
        tokens = torch.where(degraded_nodes.unsqueeze(-1), fused, tokens)

        delta = (tokens.reshape(-1, tokens.shape[-1]) - encoded.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        decoded = self._chunked(self.decoder, encoded + delta)
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

