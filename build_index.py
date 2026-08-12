"""Build the offline cave-organized DINO-v2 knowledge base."""

import argparse

from dunhuang_taco.retrieval import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Cave-organized reference image root")
    parser.add_argument("--output", required=True, help="Output .pt index")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.images, args.output, args.model, args.batch_size, args.device)
