"""Hierarchical Swin backbone used by the original ``xiufu-dan`` network.

The architecture follows ``code/xiufu-dan/model/swin_encoder.py`` and
``code/xiufu-dan/model/network.py``: four encoder stages with convolutional
patch merging and a symmetric skip-connected decoder with convolutional patch
expansion.  It is implemented with stock PyTorch so the compact reference
repository does not depend on timm or einops.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, c)


def _window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    b = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.view(
        b, height // window_size, width // window_size, window_size, window_size, -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, height, width, -1)


class WindowAttention(nn.Module):
    """Window attention with the relative-position bias used in ``xiufu-dan``."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coordinates = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += window_size - 1
        relative[..., 1] += window_size - 1
        relative[..., 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(
                batch_windows // windows, windows, self.num_heads, tokens, tokens
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.softmax(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(x)


class HierarchicalSwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: int,
        num_heads: int,
        window_size: int,
        shift: bool,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_size = min(window_size, resolution)
        self.shift_size = self.window_size // 2 if shift and resolution > self.window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.register_buffer("attention_mask", self._make_mask(), persistent=False)

    def _make_mask(self) -> Tensor | None:
        if not self.shift_size:
            return None
        size = self.resolution
        mask = torch.zeros(1, size, size, 1)
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        label = 0
        for height in slices:
            for width in slices:
                mask[:, height, width, :] = label
                label += 1
        mask = _window_partition(mask, self.window_size).squeeze(-1)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        if channels != self.dim or height != self.resolution or width != self.resolution:
            raise ValueError(
                f"expected [B,{self.dim},{self.resolution},{self.resolution}], got {tuple(x.shape)}"
            )
        tokens = x.permute(0, 2, 3, 1)
        shortcut = tokens
        tokens = self.norm1(tokens)
        if self.shift_size:
            tokens = torch.roll(tokens, (-self.shift_size, -self.shift_size), (1, 2))
        windows = _window_partition(tokens, self.window_size)
        windows = self.attn(windows, self.attention_mask)
        tokens = _window_reverse(windows, self.window_size, height, width)
        if self.shift_size:
            tokens = torch.roll(tokens, (self.shift_size, self.shift_size), (1, 2))
        tokens = shortcut + tokens
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 3, 1, 2).contiguous()


class PatchMergingConv(nn.Module):
    """Residual convolutional Patch Merging used by the xiufu-dan backbone.

    The main branch applies two 3x3 convolutions before 2x average pooling,
    while the shortcut uses a 1x1 projection followed by the same pooling.
    Their sum halves the spatial resolution and doubles the channel width.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.AvgPool2d(2, 2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


class PatchExpandConv(nn.Module):
    """Residual convolutional Patch Expansion used by the xiufu-dan decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, output_padding=1),
        )
        self.shortcut = nn.ConvTranspose2d(
            in_channels, out_channels, 3, 2, 1, output_padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


class ConvolutionBlock(nn.Module):
    """One U-Net-style 3x3 convolution used in a controlled stage replacement."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class MambaStageBlock(nn.Module):
    """Four-directional state-space block used only inside a hierarchy stage."""

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
        state = projected.new_zeros(projected.shape[0], projected.shape[2], self.state_dim)
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
            values.append((state * c_step.unsqueeze(1)).sum(dim=-1) + self.D * x_step)
        return torch.stack(values, dim=1) * F.silu(gate)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        columns = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        rows = self._scan(rows) + self._scan(rows.flip(1)).flip(1)
        columns = self._scan(columns) + self._scan(columns.flip(1)).flip(1)
        inner = rows.shape[-1]
        rows = rows.reshape(batch, height, width, inner)
        columns = columns.reshape(batch, width, height, inner).permute(0, 2, 1, 3)
        mixed = 0.25 * (rows + columns)
        return x + self.out_proj(mixed).permute(0, 3, 1, 2).contiguous()


def _head_schedule(dim: int, heads: int | tuple[int, ...] | None) -> tuple[int, ...]:
    if heads is None:
        if dim == 96:
            return (3, 6, 12, 24)
        return (4, 4, 4, 4)
    if isinstance(heads, int):
        return (heads,) * 4
    if len(heads) != 4:
        raise ValueError("heads must contain four stage values")
    return tuple(heads)


def _stage(
    kind: str,
    dim: int,
    resolution: int,
    depth: int,
    heads: int,
    window: int,
) -> nn.Sequential:
    if kind == "swin":
        blocks = (
            HierarchicalSwinBlock(dim, resolution, heads, window, shift=bool(index % 2))
            for index in range(depth)
        )
    elif kind == "unet":
        blocks = (ConvolutionBlock(dim) for _ in range(depth))
    elif kind == "mamba":
        blocks = (MambaStageBlock(dim) for _ in range(depth))
    else:
        raise ValueError(f"unknown stage kind: {kind}")
    return nn.Sequential(*blocks)


class HierarchicalEncoder(nn.Module):
    """Shared four-stage encoder; only the stage block type is replaceable."""

    def __init__(
        self,
        dim: int = 96,
        depth: int = 2,
        heads: int | tuple[int, ...] | None = None,
        patch_size: int = 256,
        kind: str = "swin",
    ):
        super().__init__()
        self.output_dim = dim * 8
        channels = (dim, dim * 2, dim * 4, dim * 8)
        resolutions = (patch_size // 4, patch_size // 8, patch_size // 16, patch_size // 32)
        head_values = _head_schedule(dim, heads)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, 4, 4),
            # PatchEmbed in xiufu-dan applies LayerNorm to the token dimension.
        )
        self.patch_norm = nn.LayerNorm(dim)
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        for index, (channel, resolution, num_heads) in enumerate(
            zip(channels, resolutions, head_values)
        ):
            window = 16 if index == 0 else 8
            self.stages.append(
                _stage(kind, channel, resolution, depth, num_heads, window)
            )
            if index < 3:
                self.mergers.append(PatchMergingConv(channel, channels[index + 1]))

    def forward(self, patches: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        x = self.patch_embed(patches)
        x = self.patch_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        skips = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            skips.append(x)
            if index < len(self.mergers):
                x = self.mergers[index](x)
        return x, tuple(skips)


class HierarchicalDecoder(nn.Module):
    """Shared skip-connected decoder; only the stage block type is replaceable."""

    def __init__(
        self,
        dim: int = 96,
        depth: int = 2,
        heads: int | tuple[int, ...] | None = None,
        patch_size: int = 256,
        kind: str = "swin",
    ):
        super().__init__()
        encoder_heads = _head_schedule(dim, heads)
        channels = (dim * 16, dim * 8, dim * 4, dim * 2)
        resolutions = (patch_size // 32, patch_size // 16, patch_size // 8, patch_size // 4)
        decoder_heads = tuple(reversed(encoder_heads))
        self.stages = nn.ModuleList()
        self.expanders = nn.ModuleList()
        for channel, resolution, num_heads in zip(channels, resolutions, decoder_heads):
            self.stages.append(_stage(kind, channel, resolution, depth, num_heads, 8))
            self.expanders.append(PatchExpandConv(channel, channel // 4))
        self.final_expand = PatchExpandConv(dim // 2, dim // 8)
        self.output = nn.Sequential(
            nn.LeakyReLU(0.1), nn.ReflectionPad2d(1), nn.Conv2d(dim // 8, 3, 3), nn.Tanh()
        )

    def forward(self, features: Tensor, skips: tuple[Tensor, ...]) -> Tensor:
        if len(skips) != 4:
            raise ValueError("the xiufu-dan Swin decoder requires four encoder skip maps")
        x = features
        for stage, expand, skip in zip(self.stages, self.expanders, reversed(skips)):
            x = stage(torch.cat((x, skip), dim=1))
            x = expand(x)
        return self.output(self.final_expand(x))


class HierarchicalSwinEncoder(HierarchicalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="swin")


class HierarchicalSwinDecoder(HierarchicalDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="swin")


class HierarchicalUNetEncoder(HierarchicalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="unet")


class HierarchicalUNetDecoder(HierarchicalDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="unet")


class HierarchicalMambaEncoder(HierarchicalEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="mamba")


class HierarchicalMambaDecoder(HierarchicalDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kind="mamba")
