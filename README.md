# Dunhuang-TACO

Official PyTorch implementation of **Dunhuang-TACO: Theme-Aware Coherent
Inpainting for Dunhuang Murals**.

## Network Architecture

![Dunhuang-TACO network architecture](assets/network_architecture.png)

Dunhuang-TACO partitions a degraded high-resolution mural into non-overlapping
patches and processes them with a shared Swin Transformer encoder. The encoded
features are divided into degraded and intact context tokens and enhanced by two
parallel branches:

- **Sparse Context Aggregation (SCA)** builds an intra-mural sparse graph to
  propagate long-range structural cues from intact regions.
- **RAG-based Theme-Aware Pipeline (R-TAP)** uses DINO-v2 to retrieve a
  thematically relevant mural and injects its reference features through a
  cross-mural sparse graph.

The two enhanced representations are fused and decoded by the shared Swin
Transformer decoder to produce the restored mural.

For the backbone ablation, `--backbone {swin,unet,mamba}` replaces only the
shared patch encoder and decoder while keeping R-TAP, SCA, and fusion fixed.
Parameter counts and computation at 1024×1024 resolution can be reproduced with
`python complexity.py --resolution 1024`. The script reports multiply-accumulate
operations (MACs); double the values if FLOPs are defined as separate multiply
and addition operations.
