"""Restore one degraded mural with Dunhuang-TACO."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from dunhuang_taco.data import load_mask, pad_to_multiple
from dunhuang_taco.model import DunhuangTACO
from dunhuang_taco.retrieval import DinoV2Retriever, load_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True, help="White denotes degradation")
    parser.add_argument("--cave", default=None, help="Known cave ID; omit for global retrieval")
    parser.add_argument("--references", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image = load_rgb(args.image)
    mask = load_mask(args.mask)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(0), image.shape[-2:], mode="nearest").squeeze(0)
    image, original_size = pad_to_multiple(image)
    mask, _ = pad_to_multiple(mask, value=0.0)
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    retriever = DinoV2Retriever(args.index, args.references, args.device)
    reference, selected = retriever.retrieve(image * (1.0 - mask), args.cave)
    if isinstance(reference, list):
        reference = reference[0].unsqueeze(0)
    reference = reference.to(device)

    model = DunhuangTACO(patch_batch_size=args.patch_batch_size).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    with torch.inference_mode():
        restored = model(image, mask, reference).completed[0]

    height, width = original_size
    restored = restored[:, :height, :width].add(1).div(2).clamp(0, 1).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(restored).save(output)
    print(f"reference={selected[0]}")
    print(f"output={output}")


if __name__ == "__main__":
    main()

