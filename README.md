# Dunhuang-TACO

Minimal PyTorch reference implementation of **Dunhuang-TACO: Theme-Aware
Coherent Inpainting for Dunhuang Murals**.

The implementation follows the method described in the manuscript:

- arbitrary high-resolution inputs are split into non-overlapping `256 x 256`
  patches and processed in chunks of 16 by a shared Swin encoder/decoder;
- SCA aggregates the Top-50% most related intact tokens inside the query mural;
- R-TAP retrieves one mural with frozen DINO-v2 features, searching the known
  cave first and falling back to the global repository, then applies the same
  Top-50% sparse graph operation to its tokens;
- SCA and R-TAP run in parallel and are combined by a learnable gate computed
  from each patch's degradation ratio;
- training uses all-pixel L1, LSGAN, and frozen ResNet-50 perceptual losses with
  weights `20`, `1`, and `15`, AdamW (`beta=(0.9, 0.999)`, `lr=1e-4`), and cosine
  annealing.

This repository contains only the proposed method. Baseline implementations,
datasets, feature indexes, and trained weights are not redistributed.

## Installation

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
python -m pip install -e .
```

The first indexing/training run downloads `facebook/dinov2-base`; the first
training run also downloads the ImageNet-pretrained ResNet-50 used by the
perceptual loss.

## Data layout

Keep the cave name as the immediate parent directory:

```text
data/
  train/
    Cave9/*.jpg
    Cave14/*.jpg
    ...
  references/
    Cave9/*.jpg
    Cave14/*.jpg
    ...
```

Images may have different resolutions. Each height and width is reflect-padded
to the next multiple of 256, and the original size is restored at export.
Training uses a batch size of one mural; `--patch-batch-size 16` is the
chunk-wise patch mini-batch reported in the paper.

## 1. Build the DINO-v2 reference index

```bash
python build_index.py \
  --images data/references \
  --output data/reference_index.pt
```

The index contains DINO-v2 features and relative image paths. Its contents are
exactly the images supplied through `--images`.

## 2. Train

```bash
python train.py \
  --images data/train \
  --references data/references \
  --index data/reference_index.pt \
  --output checkpoints/dunhuang_taco \
  --epochs 100 \
  --patch-batch-size 16
```

Free-form masks with ratios sampled between 1% and 60% are generated online.
Checkpoints store only the proposed generator, discriminator, optimizers, and
schedulers.

## 3. Restore a mural

The mask is a grayscale image in which white (`1`) denotes degradation and
black (`0`) denotes intact content.

```bash
python infer.py \
  --image examples/degraded.png \
  --mask examples/mask.png \
  --cave Cave85 \
  --references data/references \
  --index data/reference_index.pt \
  --checkpoint checkpoints/dunhuang_taco/final.pt \
  --output outputs/restored.png
```

Omit `--cave` when the cave identifier is unknown; R-TAP then searches the full
index. The supplied image is multiplied by `(1 - mask)` before encoding, as in
the manuscript.

## Main files

- `dunhuang_taco/model.py`: shared Swin patch encoder/decoder, SCA, R-TAP graph,
  and mask-guided fusion.
- `dunhuang_taco/retrieval.py`: frozen DINO-v2 indexing and cave-aware retrieval.
- `dunhuang_taco/losses.py`: L1, LSGAN, and frozen ResNet-50 perceptual losses.
- `dunhuang_taco/data.py`: mural loading and free-form mask generation.
- `build_index.py`, `train.py`, `infer.py`: reproducible command-line entry points.

## Notes

- Old PIC/ADE20K modules and checkpoints are intentionally excluded: they do
  not implement the architecture described in the revised manuscript and are
  not compatible with this model.
- The code expects RGB inputs in `[-1, 1]`, while masks use `{0, 1}` with
  `1 = degraded` throughout.

