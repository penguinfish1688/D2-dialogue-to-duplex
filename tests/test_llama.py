import pytest
import torch
from transformers import __version__

pytestmark = pytest.mark.skipif(not __version__.startswith("4."), reason="Llama environment only")


def decoder():
    from transformers import Qwen2Config, Qwen2Model

    torch.manual_seed(19)
    model = Qwen2Model(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_dropout=0,
            max_position_embeddings=128,
        )
    )
    return model.eval().requires_grad_(False)


@torch.no_grad()
def test_captured_decoder_math_matches_native_prefill_and_serial_decode():
    from d2_llama.inference.fixed_kv import GraphDecoder

    model = decoder()
    graph = GraphDecoder(model, capacity=48, capture=False)
    inputs = model.embed_tokens(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
    reference = model(inputs_embeds=inputs, use_cache=True, return_dict=True).last_hidden_state
    actual = torch.cat(
        [graph.step(inputs[:, :5]), graph.step(inputs[:, 5:6]), graph.step(inputs[:, 6:])], dim=1
    )
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    pointers = [tensor.data_ptr() for tensors in graph.storage for tensor in tensors]
    graph.reset()
    graph.step(inputs)
    assert pointers == [tensor.data_ptr() for tensors in graph.storage for tensor in tensors]
    with pytest.raises(ValueError, match="budget"):
        graph.step(inputs.repeat(1, 8, 1))


@torch.no_grad()
def test_unwritten_cache_slots_cannot_affect_output():
    from d2_llama.inference.fixed_kv import GraphDecoder

    model = decoder()
    graph = GraphDecoder(model, capacity=16, capture=False)
    inputs = model.embed_tokens(torch.tensor([[1, 2, 3]]))
    expected = graph.step(inputs)
    graph.reset()
    for tensors in graph.storage:
        for tensor in tensors:
            tensor.fill_(1000)
    actual = graph.step(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interruption_discards_pending_pcm():
    from d2_llama.inference.session import PCMQueue

    queue = PCMQueue(b"\1\0" * 10)
    assert queue.pop_exact_or_silence(4) == b"\1\0" * 4
    queue.clear()
    assert queue.pop_exact_or_silence(8) == bytes(16)


def test_macro_audio_mask_respects_visibility():
    from d2_llama.model.encoder import block_causal_mask

    for k in (1, 2, 4, 8):
        rows = 5 * k
        mask = block_causal_mask(2 * rows, k)
        assert torch.isfinite(mask[:rows, :rows]).all()
        assert torch.isneginf(mask[:rows, rows:]).all()
        assert torch.isfinite(mask[rows:, :]).all()


def test_renderer_chunk_mask_boundaries_and_left_context():
    from d2_llama.inference.mask import subsequent_chunk_mask

    mask = subsequent_chunk_mask(10, 3, num_left_chunks=1)
    assert mask.is_contiguous() and mask.dtype == torch.bool
    assert mask[0].tolist() == [True] * 3 + [False] * 7
    assert mask[5].tolist() == [True] * 6 + [False] * 4
    assert mask[6].tolist() == [False] * 3 + [True] * 6 + [False]
    assert mask[9].tolist() == [False] * 6 + [True] * 4
    assert subsequent_chunk_mask(10, 3)[9].all()
