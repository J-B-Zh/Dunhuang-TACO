"""Dunhuang-TACO generator and discriminator.

The implementation mirrors Sections 3.2--3.5 of the manuscript: a shared
chunk-wise Swin patch backbone, parallel SCA/R-TAP sparse graphs, and
mask-guided convex fusion. The patch encoder/decoder can be replaced with
Swin Transformer, U-Net-style CNN, or Mamba-style state-space blocks for the
backbone ablation while keeping SCA, R-TAP, and feature fusion unchanged.
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


class DoubleConv(nn.Module):
    """Two convolutional layers used at each U-Net resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetPatchEncoder(nn.Module):
    """Two-level U-Net encoder returning its skip features explicitly."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        del depth, heads
        base = max(16, dim // 2)
        self.stage1 = DoubleConv(3, base)
        self.stage2 = DoubleConv(base, dim)
        self.bottleneck = DoubleConv(dim, dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, patches: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        skip1 = self.stage1(patches)
        skip2 = self.stage2(self.pool(skip1))
        encoded = self.bottleneck(self.pool(skip2))
        return encoded, (skip1, skip2)


class UNetPatchDecoder(nn.Module):
    """U-Net decoder with skip connections from :class:`UNetPatchEncoder`."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        del depth, heads
        base = max(16, dim // 2)
        self.up1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.stage1 = DoubleConv(dim * 2, dim)
        self.up2 = nn.ConvTranspose2d(dim, base, kernel_size=2, stride=2)
        self.stage2 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 3, 1)

    def forward(self, features: Tensor, skips: tuple[Tensor, Tensor]) -> Tensor:
        skip1, skip2 = skips
        x = self.stage1(torch.cat((self.up1(features), skip2), dim=1))
        x = self.stage2(torch.cat((self.up2(x), skip1), dim=1))
        return torch.tanh(self.output(x))


class MambaBlock(nn.Module):
    """Pure-PyTorch selective state-space block for backbone ablations.

    The recurrent scan is linear in sequence length. Four directional scans
    (left-to-right, right-to-left, top-to-bottom, and bottom-to-top) provide
    bidirectional 2-D context without requiring a platform-specific CUDA
    extension.
    """

    def __init__(
        self, dim: int, expansion: int = 2, conv_kernel: int = 3, state_dim: int = 16
    ):
        super().__init__()
        inner = dim * expansion
        dt_rank = max(1, math.ceil(dim / 16))
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.depthwise = nn.Conv1d(
            inner, inner, conv_kernel, padding=conv_kernel - 1, groups=inner
        )
        self.x_proj = nn.Linear(inner, dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(inner, 1)
        )
        self.D = nn.Parameter(torch.ones(inner))
        self.dt_rank = dt_rank
        self.state_dim = state_dim
        self.out_proj = nn.Linear(inner, dim)

    def _scan(self, x: Tensor) -> Tensor:
        projected, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        projected = self.depthwise(projected.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        projected = F.silu(projected)
        parameters = self.x_proj(projected)
        dt, b_vector, c_vector = torch.split(
            parameters, (self.dt_rank, self.state_dim, self.state_dim), dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        transition = -torch.exp(self.A_log).to(projected.dtype)
        state = projected.new_zeros(
            projected.shape[0], projected.shape[2], self.state_dim
        )
        values = []
        for x_step, dt_step, b_step, c_step in zip(
            projected.unbind(dim=1),
            dt.unbind(dim=1),
            b_vector.unbind(dim=1),
            c_vector.unbind(dim=1),
        ):
            discrete_a = torch.exp(dt_step.unsqueeze(-1) * transition)
            discrete_b = dt_step.unsqueeze(-1) * b_step.unsqueeze(1)
            state = discrete_a * state + discrete_b * x_step.unsqueeze(-1)
            y_step = (state * c_step.unsqueeze(1)).sum(dim=-1) + self.D * x_step
            values.append(y_step)
        return torch.stack(values, dim=1) * F.silu(gate)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        cols = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        rows_bi = self._scan(rows) + self._scan(rows.flip(1)).flip(1)
        cols_bi = self._scan(cols) + self._scan(cols.flip(1)).flip(1)
        rows_bi = rows_bi.reshape(b, h, w, c)
        cols_bi = cols_bi.reshape(b, w, h, c).permute(0, 2, 1, 3)
        mixed = 0.25 * (rows_bi + cols_bi)
        return x + self.out_proj(mixed).permute(0, 3, 1, 2).contiguous()


class MambaPatchEncoder(nn.Module):
    """Mamba-style state-space replacement for the Swin patch encoder."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        del heads
        self.embed = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*(MambaBlock(dim) for _ in range(depth)))

    def forward(self, patches: Tensor) -> Tensor:
        return self.blocks(self.embed(patches))


class MambaPatchDecoder(nn.Module):
    """Mamba-style decoder paired with :class:`MambaPatchEncoder`."""

    def __init__(self, dim: int = 96, depth: int = 4, heads: int = 4):
        super().__init__()
        del heads
        self.blocks = nn.Sequential(*(MambaBlock(dim) for _ in range(depth)))
        self.output = nn.Sequential(nn.Conv2d(dim, 3 * 16, 3, padding=1), nn.PixelShuffle(4))

    def forward(self, features: Tensor) -> Tensor:
        return torch.tanh(self.output(self.blocks(features)))


BACKBONES: dict[str, tuple[type[nn.Module], type[nn.Module]]] = {
    "swin": (SwinPatchEncoder, SwinPatchDecoder),
    "unet": (UNetPatchEncoder, UNetPatchDecoder),
    "mamba": (MambaPatchEncoder, MambaPatchDecoder),
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
        depth: int = 4,
        heads: int = 4,
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
        self.encoder = encoder_type(dim, depth, heads)
        self.decoder = decoder_type(dim, depth, heads)
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

    def _chunked_encode(
        self, values: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        maps = []
        skip1 = []
        skip2 = []
        for start in range(0, len(values), self.patch_batch_size):
            encoded = self.encoder(values[start : start + self.patch_batch_size])
            if isinstance(encoded, tuple):
                feature_map, skips = encoded
                maps.append(feature_map)
                skip1.append(skips[0])
                skip2.append(skips[1])
            else:
                maps.append(encoded)
        return torch.cat(maps, dim=0), (
            (torch.cat(skip1, dim=0), torch.cat(skip2, dim=0)) if skip1 else None
        )

    def _chunked_decode(
        self, values: Tensor, skips: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        outputs = []
        for start in range(0, len(values), self.patch_batch_size):
            chunk = values[start : start + self.patch_batch_size]
            if skips is None:
                outputs.append(self.decoder(chunk))
            else:
                skip_chunk = (
                    skips[0][start : start + self.patch_batch_size],
                    skips[1][start : start + self.patch_batch_size],
                )
                outputs.append(self.decoder(chunk, skip_chunk))
        return torch.cat(outputs, dim=0)

    def _encode_tokens(
        self, image: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None, Tensor, int, int]:
        patches, gh, gw = self._split(image)
        maps, skips = self._chunked_encode(patches)
        tokens = maps.mean(dim=(-2, -1)).view(image.shape[0], gh * gw, -1)
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

        delta = (tokens.reshape(-1, tokens.shape[-1]) - encoded.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
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
