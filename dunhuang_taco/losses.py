"""Training objectives specified in Section 3.6 of the manuscript."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50PerceptualLoss(nn.Module):
    """All-element normalized L1 distance on frozen ResNet-50 feature maps."""

    def __init__(self):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.stages = nn.ModuleList([model.layer1, model.layer2, model.layer3])
        self.requires_grad_(False).eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def train(self, mode: bool = True) -> "ResNet50PerceptualLoss":
        return super().train(False)

    def _features(self, image: Tensor) -> list[Tensor]:
        x = (image.add(1).div(2) - self.mean) / self.std
        x = self.stem(x)
        values = []
        for stage in self.stages:
            x = stage(x)
            values.append(x)
        return values

    def forward(self, generated: Tensor, ground_truth: Tensor) -> Tensor:
        generated_features = self._features(generated)
        with torch.no_grad():
            ground_truth_features = self._features(ground_truth)
        return sum(F.l1_loss(a, b) for a, b in zip(generated_features, ground_truth_features))


def discriminator_lsgan(discriminator: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    return (discriminator(real) - 1).square().mean() + discriminator(fake.detach()).square().mean()


def generator_objective(
    discriminator: nn.Module,
    perceptual: nn.Module,
    generated: Tensor,
    ground_truth: Tensor,
    lambda_rec: float = 20.0,
    lambda_adv: float = 1.0,
    lambda_perc: float = 15.0,
) -> dict[str, Tensor]:
    reconstruction = F.l1_loss(generated, ground_truth)
    adversarial = (discriminator(generated) - 1).square().mean()
    perceptual_loss = perceptual(generated, ground_truth)
    overall = lambda_rec * reconstruction + lambda_adv * adversarial + lambda_perc * perceptual_loss
    return {
        "overall": overall,
        "reconstruction": reconstruction,
        "adversarial": adversarial,
        "perceptual": perceptual_loss,
    }

