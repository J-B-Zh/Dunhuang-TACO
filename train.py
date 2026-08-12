"""Train Dunhuang-TACO with the settings reported in the manuscript."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dunhuang_taco.data import ThemeDHDataset
from dunhuang_taco.losses import ResNet50PerceptualLoss, discriminator_lsgan, generator_objective
from dunhuang_taco.model import DunhuangTACO, PatchDiscriminator
from dunhuang_taco.retrieval import DinoV2Retriever


def single_mural(samples: list[dict[str, object]]) -> dict[str, object]:
    """Keep variable-resolution murals unstacked for a loader batch of one."""
    return samples[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Cave-organized training murals")
    parser.add_argument("--references", required=True, help="Cave-organized reference murals")
    parser.add_argument("--index", required=True, help="DINO-v2 reference index")
    parser.add_argument("--output", required=True, help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: DunhuangTACO,
    discriminator: PatchDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler.LRScheduler,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dataset = ThemeDHDataset(args.images)
    # One mural per loader batch; its 256x256 patches are processed in chunks of 16.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=single_mural,
    )
    retriever = DinoV2Retriever(args.index, args.references, args.device)

    generator = DunhuangTACO(patch_batch_size=args.patch_batch_size).to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual = ResNet50PerceptualLoss().to(device)
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.epochs)
    output_dir = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        for step, batch in enumerate(loader, start=1):
            ground_truth = batch["image"].unsqueeze(0).to(device, non_blocking=True)
            mask = batch["mask"].unsqueeze(0).to(device, non_blocking=True)
            cave = batch["cave"]
            masked = ground_truth * (1.0 - mask)
            reference, _ = retriever.retrieve(masked, cave)
            if isinstance(reference, list):
                reference = reference[0].unsqueeze(0)
            reference = reference.to(device, non_blocking=True)

            generated = generator(ground_truth, mask, reference).completed

            optimizer_d.zero_grad(set_to_none=True)
            loss_d = discriminator_lsgan(discriminator, ground_truth, generated)
            loss_d.backward()
            optimizer_d.step()

            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            losses = generator_objective(discriminator, perceptual, generated, ground_truth)
            losses["overall"].backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)

            if step % 20 == 0 or step == len(loader):
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                    f"g={losses['overall'].item():.4f} d={loss_d.item():.4f} "
                    f"rec={losses['reconstruction'].item():.4f} "
                    f"adv={losses['adversarial'].item():.4f} perc={losses['perceptual'].item():.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                scheduler_g,
                scheduler_d,
            )

    save_checkpoint(
        output_dir / "final.pt",
        args.epochs,
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
    )


if __name__ == "__main__":
    main()

