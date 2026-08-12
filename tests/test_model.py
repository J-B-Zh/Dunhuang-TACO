"""CPU smoke tests that do not download pretrained weights."""

import torch

from dunhuang_taco.model import DunhuangTACO, MaskGuidedFusion, SparseGraphAggregation


def test_sparse_graph_preserves_shape_and_uses_half_valid_neighbors() -> None:
    graph = SparseGraphAggregation(dim=8, topk_ratio=0.5)
    queries = torch.randn(2, 3, 8)
    keys = torch.randn(2, 6, 8)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    assert graph(queries, keys, valid).shape == queries.shape


def test_mask_gate_is_a_convex_combination() -> None:
    fusion = MaskGuidedFusion()
    cross = torch.ones(1, 4, 8)
    intra = torch.zeros_like(cross)
    fused, gate = fusion(cross, intra, torch.linspace(0, 1, 4).view(1, 4))
    assert fused.shape == cross.shape
    assert gate.shape == (1, 4, 1)
    assert torch.all((gate > 0) & (gate < 1))
    assert torch.allclose(fused, gate.expand_as(fused))


def test_minimal_forward_and_known_pixels() -> None:
    model = DunhuangTACO(patch_batch_size=2, dim=16, depth=2, heads=4)
    image = torch.randn(1, 3, 256, 512).clamp(-1, 1)
    mask = torch.zeros(1, 1, 256, 512)
    mask[:, :, 64:192, 256:512] = 1
    reference = torch.randn_like(image).clamp(-1, 1)
    output = model(image, mask, reference)
    assert output.completed.shape == image.shape
    assert output.gate.shape == (1, 2)
    assert torch.equal(output.completed * (1 - mask), image * (1 - mask))


def test_all_backbones_preserve_the_generator_contract() -> None:
    image = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 64:192, 64:192] = 1
    reference = torch.randn_like(image).clamp(-1, 1)
    for backbone in ("swin", "unet", "mamba"):
        model = DunhuangTACO(
            backbone=backbone, patch_batch_size=1, dim=16, depth=1, heads=4
        )
        output = model(image, mask, reference)
        assert output.completed.shape == image.shape
        assert torch.equal(output.completed * (1 - mask), image * (1 - mask))


def test_default_swin_matches_xiufu_dan_hierarchy() -> None:
    model = DunhuangTACO()
    assert [len(stage) for stage in model.encoder.stages] == [2, 2, 2, 2]
    assert [stage[0].dim for stage in model.encoder.stages] == [96, 192, 384, 768]
    assert [stage[0].attn.num_heads for stage in model.encoder.stages] == [3, 6, 12, 24]
    assert [stage[0].window_size for stage in model.encoder.stages] == [16, 8, 8, 8]
    assert [len(stage) for stage in model.decoder.stages] == [2, 2, 2, 2]
    assert [stage[0].dim for stage in model.decoder.stages] == [1536, 768, 384, 192]
    assert model.feature_dim == 768
    assert model.graph_dim == 768


def test_backbone_ablation_replaces_only_stage_blocks() -> None:
    models = {name: DunhuangTACO(backbone=name) for name in ("swin", "unet", "mamba")}
    for model in models.values():
        assert [len(stage) for stage in model.encoder.stages] == [2, 2, 2, 2]
        assert [len(stage) for stage in model.decoder.stages] == [2, 2, 2, 2]
        assert [tuple(layer.shortcut[0].weight.shape) for layer in model.encoder.mergers] == [
            (192, 96, 1, 1), (384, 192, 1, 1), (768, 384, 1, 1)
        ]
        assert [tuple(layer.shortcut.weight.shape) for layer in model.decoder.expanders] == [
            (1536, 384, 3, 3), (768, 192, 3, 3),
            (384, 96, 3, 3), (192, 48, 3, 3)
        ]
