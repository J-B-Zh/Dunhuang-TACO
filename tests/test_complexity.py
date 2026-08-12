"""Checks for the deterministic backbone-complexity report."""

from dunhuang_taco.model import DunhuangTACO
from complexity import backbone_macs, graph_macs, trainable_parameters


def test_backbone_complexities_are_positive_and_distinct() -> None:
    macs = {}
    parameters = {}
    for name in ("swin", "unet", "mamba"):
        model = DunhuangTACO(backbone=name)
        encoder_macs, decoder_macs = backbone_macs(name, 256, 96, 2)
        macs[name] = encoder_macs + decoder_macs
        parameters[name] = trainable_parameters(model.encoder) + trainable_parameters(model.decoder)

    assert all(value > 0 for value in macs.values())
    assert all(value > 0 for value in parameters.values())
    assert len(set(macs.values())) == 3
    assert len(set(parameters.values())) == 3
    assert graph_macs(patch_count=16, dim=96) > 0
