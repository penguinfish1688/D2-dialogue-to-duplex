import pytest
import torch
from transformers import __version__

pytestmark = pytest.mark.skipif(not __version__.startswith("5."), reason="Qwen environment only")


def test_all_macro_layouts_preserve_native_order():
    from d2_qwen.model.duplex import pack_macro_triplets

    for k in (1, 2, 4, 8, 13):
        environment = torch.arange(2 * k).reshape(1, 2 * k, 1)
        own = environment + 100
        text = environment + 200
        actual = pack_macro_triplets(environment, own, text, k).flatten().tolist()
        expected = []
        for offset in (0, k):
            expected.extend(range(offset, offset + k))
            expected.extend(range(100 + offset, 100 + offset + k))
            expected.extend(range(200 + offset, 200 + offset + k))
        assert actual == expected


def test_causal_encoder_mask_cannot_see_future_macro():
    from d2_qwen.model.encoder import plan_macro_attention_blocks, macro_attention_block_mask

    for k in (1, 2, 4, 8, 13):
        blocks = plan_macro_attention_blocks(2 * k, frames_per_unit=k, window_tokens=104)
        mask = macro_attention_block_mask(
            blocks[0], frames_per_unit=k, window_tokens=104, device="cpu"
        )
        assert mask[:k, :k].all()
        assert not mask[:k, k:].any()
        assert mask[k:, :].all()
