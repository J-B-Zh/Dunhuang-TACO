"""Frozen DINO-v2 knowledge-base construction and R-TAP retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from transformers import AutoImageProcessor, Dinov2Model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return tensor.mul_(2.0).sub_(1.0)


class DinoV2Encoder(nn.Module):
    """Frozen global semantic encoder used both offline and online."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | torch.device | None = None):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.backbone.requires_grad_(False).eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone.to(self.device)

    @torch.inference_mode()
    def forward(self, images: Tensor | Iterable[Image.Image]) -> Tensor:
        if isinstance(images, Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            pil_images = [to_pil_image(image.add(1).div(2).clamp(0, 1).cpu()) for image in images]
        else:
            pil_images = list(images)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.backbone(**inputs).last_hidden_state[:, 0]
        return F.normalize(features, dim=-1)


def build_index(
    images_root: str | Path,
    output: str | Path,
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 16,
    device: str | None = None,
) -> None:
    """Extract one DINO-v2 CLS feature per repository image."""
    root = Path(images_root).resolve()
    paths = find_images(root)
    if not paths:
        raise ValueError(f"No reference images found under {root}")
    encoder = DinoV2Encoder(model_name, device)
    rows: list[Tensor] = []
    for start in range(0, len(paths), batch_size):
        batch: list[Image.Image] = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                batch.append(image.convert("RGB").copy())
        rows.append(encoder(batch).cpu())

    relative = [path.relative_to(root).as_posix() for path in paths]
    archive = {
        "model_name": model_name,
        "features": torch.cat(rows),
        "paths": relative,
        "caves": [Path(path).parent.name for path in relative],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(archive, output)


class DinoV2Retriever:
    """Retrieve the most similar reference within a cave or globally."""

    def __init__(
        self,
        index_path: str | Path,
        images_root: str | Path,
        device: str | None = None,
    ):
        archive = torch.load(index_path, map_location="cpu", weights_only=True)
        self.features: Tensor = F.normalize(archive["features"].float(), dim=-1)
        self.paths: list[str] = list(archive["paths"])
        self.caves: list[str] = list(archive["caves"])
        self.images_root = Path(images_root)
        self.encoder = DinoV2Encoder(archive["model_name"], device)

    @torch.inference_mode()
    def retrieve(self, degraded: Tensor, cave_id: str | list[str] | tuple[str, ...] | None = None) -> tuple[Tensor | list[Tensor], list[str]]:
        """Return one reference per query and its repository-relative path.

        A supplied cave identifier strictly constrains retrieval to that cave.
        When it is unavailable (``None``), all repository images are searched.
        """
        if degraded.ndim == 3:
            degraded = degraded.unsqueeze(0)
        query = self.encoder(degraded).cpu()
        references: list[Tensor] = []
        selected_paths: list[str] = []
        cave_ids = list(cave_id) if isinstance(cave_id, (list, tuple)) else [cave_id] * len(query)
        if len(cave_ids) != len(query):
            raise ValueError("cave_id sequence must match the query batch size")
        for row, query_cave in zip(query, cave_ids):
            candidates = [i for i, cave in enumerate(self.caves) if cave == query_cave] if query_cave else list(range(len(self.paths)))
            if not candidates:
                raise ValueError(f"No reference images are indexed for cave {query_cave!r}")
            candidate_features = self.features[candidates]
            best_local = int(torch.mv(candidate_features, row).argmax())
            best = candidates[best_local]
            relative = self.paths[best]
            references.append(load_rgb(self.images_root / relative))
            selected_paths.append(relative)
        return torch.stack(references) if len({x.shape for x in references}) == 1 else references, selected_paths
