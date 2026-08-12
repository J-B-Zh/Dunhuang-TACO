"""ThemeDH image loading and online free-form mask generation."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .retrieval import find_images, load_rgb


def pad_to_multiple(image: Tensor, multiple: int = 256, value: float | None = None) -> tuple[Tensor, tuple[int, int]]:
    """Pad the bottom/right and return the unpadded ``(height, width)``."""
    height, width = image.shape[-2:]
    pad_h = math.ceil(height / multiple) * multiple - height
    pad_w = math.ceil(width / multiple) * multiple - width
    mode = "constant" if value is not None else "reflect"
    kwargs = {"value": value} if value is not None else {}
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode, **kwargs), (height, width)


def load_mask(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        mask = pil_to_tensor(image.convert("L")).float().div_(255.0)
    return (mask >= 0.5).float()


def free_form_mask(height: int, width: int, min_ratio: float = 0.01, max_ratio: float = 0.60) -> Tensor:
    """Generate irregular brush-stroke degradation with a sampled mask ratio."""
    target = random.uniform(min_ratio, max_ratio)
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    diagonal = math.sqrt(height * height + width * width)
    minimum_width = max(8, round(diagonal * 0.008))
    maximum_width = max(minimum_width + 1, round(diagonal * 0.06))
    attempts = 0
    while np.asarray(canvas, dtype=np.uint8).mean() / 255.0 < target and attempts < 256:
        points = [(random.randrange(width), random.randrange(height))]
        for _ in range(random.randint(2, 8)):
            x, y = points[-1]
            length = random.uniform(diagonal * 0.03, diagonal * 0.18)
            angle = random.uniform(0, 2 * math.pi)
            points.append(
                (
                    int(np.clip(x + length * math.cos(angle), 0, width - 1)),
                    int(np.clip(y + length * math.sin(angle), 0, height - 1)),
                )
            )
        stroke_width = random.randint(minimum_width, maximum_width)
        draw.line(points, fill=255, width=stroke_width, joint="curve")
        radius = stroke_width // 2
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        attempts += 1
    return pil_to_tensor(canvas).float().div_(255.0)


class ThemeDHDataset(Dataset):
    """Recursively load murals; the immediate parent directory is the cave ID."""

    def __init__(self, images_root: str | Path):
        self.root = Path(images_root)
        self.paths = find_images(self.root)
        if not self.paths:
            raise ValueError(f"No mural images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        image = load_rgb(path)
        mask = free_form_mask(*image.shape[-2:])
        image, original_size = pad_to_multiple(image)
        mask, _ = pad_to_multiple(mask, value=0.0)
        return {
            "image": image,
            "mask": mask,
            "cave": path.parent.name,
            "path": str(path),
            "original_size": original_size,
        }

