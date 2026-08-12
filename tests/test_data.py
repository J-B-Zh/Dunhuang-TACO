import torch

from dunhuang_taco.data import free_form_mask, pad_to_multiple


def test_padding_and_mask_convention() -> None:
    image = torch.zeros(3, 300, 500)
    padded, original = pad_to_multiple(image)
    mask = free_form_mask(300, 500, min_ratio=0.05, max_ratio=0.10)
    assert original == (300, 500)
    assert padded.shape == (3, 512, 512)
    assert mask.shape == (1, 300, 500)
    assert 0 < mask.mean() <= 1
