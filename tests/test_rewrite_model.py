from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch import nn

from drummer.rewrite_codec import (
    BOS, BYTE_BASE, COPY_BASE, EOS, MAX_COPY_SLOTS, MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS, PAD, SEP, VOCAB_SIZE, encode_target, prepare_input,
)
from drummer.rewrite_model import EncodedSource, RewriteConfig, RewriteModel


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny():
    # Component checks only, deliberately not the research architecture.
    torch.manual_seed(7)
    return RewriteModel(RewriteConfig(width=16, heads=2, feedforward=32,
                                      encoder_layers=1, decoder_layers=1))


def byte(value):
    return BYTE_BASE + ord(value)


def matrix(rows):
    width = max(len(row) for row in rows)
    return torch.tensor([list(row) + [PAD] * (width - len(row)) for row in rows], dtype=torch.long)


def source():
    return torch.tensor([[BOS, byte("x"), SEP, EOS]], dtype=torch.long)


def test_default_architecture_fixed_vocabulary_and_actual_parameter_count():
    config = RewriteConfig()
    assert config.is_research_default
    assert (config.width, config.heads, config.feedforward,
            config.encoder_layers, config.decoder_layers, config.dropout) == (256, 4, 1024, 2, 2, 0.0)
    model = RewriteModel(config)
    assert isinstance(model.transformer, nn.Transformer)
    assert len(model.transformer.encoder.layers) == len(model.transformer.decoder.layers) == 2
    assert model.transformer.encoder.layers[0].self_attn.num_heads == 4
    assert model.transformer.decoder.layers[0].linear1.out_features == 1024
    assert model.token_embedding.num_embeddings == VOCAB_SIZE == 324
    assert model.position_embedding.num_embeddings == MAX_INPUT_TOKENS
    assert model.output_head.weight is not model.token_embedding.weight
    assert model.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert model.parameter_count == 4_377_924
    assert model.parameter_count > tiny().parameter_count
    assert not tiny().config.is_research_default


@pytest.mark.parametrize("change", [
    {"width": True}, {"width": "16"}, {"width": 16.0}, {"width": 0}, {"width": 513},
    {"width": 17, "heads": 2}, {"heads": 0}, {"heads": False}, {"heads": 17},
    {"feedforward": 128}, {"feedforward": float("nan")}, {"feedforward": 4097},
    {"encoder_layers": 0}, {"encoder_layers": True}, {"decoder_layers": 5},
    {"dropout": False}, {"dropout": 0}, {"dropout": float("nan")}, {"dropout": 0.1},
])
def test_config_is_strict_bounded_and_never_coerces(change):
    with pytest.raises(ValueError):
        replace(RewriteConfig(), **change)


def test_mapping_config_is_not_silently_promoted_to_research_configuration():
    with pytest.raises(TypeError):
        RewriteModel({"width": 16})


def test_teacher_forced_forward_backward_has_finite_loss_and_gradients_ignoring_pad():
    model = tiny()
    inputs = [prepare_input('inspect "a.py"', 'ACK "b.py"'), prepare_input("x", "")]
    targets = [encode_target('read "a.py"', inputs[0]), encode_target("x", inputs[1])]
    src = matrix([prepared.tokens for prepared in inputs])
    counts = torch.tensor([len(prepared.copies) for prepared in inputs], dtype=torch.long)
    prefix = matrix([target[:-1] for target in targets])
    target = matrix([target[1:] for target in targets])
    logits = model(src, prefix, counts)
    assert logits.shape == (*prefix.shape, VOCAB_SIZE)
    loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target.flatten(), ignore_index=PAD)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.token_embedding.weight.grad.abs().sum() > 0


def test_prefix_attention_is_causal_not_future_conditioned():
    model = tiny().eval()
    memory = model.encode(source(), torch.tensor([0]))
    one = torch.tensor([[BOS, byte("a"), byte("b"), byte("c")]])
    two = torch.tensor([[BOS, byte("a"), byte("d"), byte("e")]])
    with torch.no_grad():
        first, second = model.decode(memory, one), model.decode(memory, two)
    torch.testing.assert_close(first[:, :2], second[:, :2], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(first[:, 2:, EOS], second[:, 2:, EOS])


def test_source_and_prefix_padding_do_not_change_active_outputs():
    model = tiny().eval()
    counts = torch.tensor([0])
    prefix = torch.tensor([[BOS, byte("a"), byte("b")]])
    with torch.no_grad():
        original = model(source(), prefix, counts)
        padded = model(nn.functional.pad(source(), (0, 4)), nn.functional.pad(prefix, (0, 3)), counts)
    torch.testing.assert_close(original, padded[:, :prefix.shape[1]], rtol=1e-5, atol=1e-6)


def test_output_masks_specials_quote_and_per_row_unavailable_copy_slots():
    model = tiny()
    rows = [[BOS, SEP, EOS], [BOS, COPY_BASE, COPY_BASE + 1, SEP, EOS]]
    counts = torch.tensor([0, 2])
    logits = model(matrix(rows), torch.tensor([[BOS], [BOS]]), counts)
    forbidden = [PAD, BOS, SEP, byte('"')]
    assert byte('"') == 38
    assert torch.isneginf(logits[:, :, forbidden]).all()
    assert torch.isneginf(logits[0, :, COPY_BASE:]).all()
    assert torch.isfinite(logits[1, :, COPY_BASE:COPY_BASE + 2]).all()
    assert torch.isneginf(logits[1, :, COPY_BASE + 2:]).all()
    assert torch.isfinite(logits[:, :, EOS]).all()


@pytest.mark.parametrize("row", [
    [], [BOS], [BOS, SEP], [BOS, EOS, SEP], [BOS, SEP, EOS, byte("a")],
    [BOS, BOS, SEP, EOS], [BOS, SEP, SEP, EOS], [BOS, SEP, EOS, EOS],
    [BOS, PAD, SEP, EOS], [PAD, PAD, PAD], [BOS, SEP, -1, EOS],
    [BOS, SEP, VOCAB_SIZE, EOS], [BOS, COPY_BASE, SEP, EOS],
])
def test_malformed_source_framing_and_ids_fail_before_attention(row):
    with pytest.raises(ValueError):
        tiny().encode(torch.tensor([row], dtype=torch.long), torch.tensor([0]))


@pytest.mark.parametrize("tokens", [
    torch.ones(4, dtype=torch.long), torch.ones((0, 4), dtype=torch.long),
    torch.ones((1, 4), dtype=torch.float32), torch.ones((1, 4), dtype=torch.bool),
    torch.ones((1, MAX_INPUT_TOKENS + 1), dtype=torch.long),
])
def test_source_rank_dtype_and_length_are_strict(tokens):
    with pytest.raises(ValueError):
        tiny().encode(tokens, torch.tensor([0]))


@pytest.mark.parametrize("counts", [
    torch.tensor([True]), torch.tensor([0.0]), torch.tensor(0), torch.tensor([0, 0]),
    torch.tensor([-1]), torch.tensor([MAX_COPY_SLOTS + 1]), [0],
    torch.tensor([1]),  # A declared slot must actually occur in the source.
])
def test_copy_count_shape_dtype_bounds_and_observation_are_checked(counts):
    with pytest.raises(ValueError):
        tiny().encode(source(), counts)


@pytest.mark.parametrize("row", [
    [], [PAD], [byte("a")], [BOS, BOS], [BOS, SEP], [BOS, byte('"')],
    [BOS, PAD, byte("a")], [BOS, EOS, byte("a")], [BOS, EOS, EOS],
    [BOS, COPY_BASE], [BOS, -1], [BOS, VOCAB_SIZE],
])
def test_malformed_teacher_prefix_cannot_smuggle_invalid_output_tokens(row):
    model = tiny()
    memory = model.encode(source(), torch.tensor([0]))
    with pytest.raises(ValueError):
        model.decode(memory, torch.tensor([row], dtype=torch.long))


def test_valid_terminal_eos_then_padding_is_allowed_in_teacher_prefix():
    model = tiny()
    memory = model.encode(source(), torch.tensor([0]))
    logits = model.decode(memory, torch.tensor([[BOS, byte("a"), EOS, PAD]]))
    assert torch.isfinite(logits[:, :, EOS]).all()


def test_memory_is_explicit_frozen_and_clones_caller_metadata():
    model = tiny()
    counts = torch.tensor([0])
    memory = model.encode(source(), counts)
    counts[0] = 5
    assert memory.copy_counts.tolist() == [0]
    with pytest.raises(FrozenInstanceError):
        memory.hidden = memory.hidden
    with pytest.raises(ValueError):
        model.decode(replace(memory, hidden=memory.hidden.double()), torch.tensor([[BOS]]))
    with pytest.raises(ValueError):
        model.decode(replace(memory, padding=memory.padding.long()), torch.tensor([[BOS]]))
    with pytest.raises(ValueError):
        model.decode(replace(memory, padding=torch.ones_like(memory.padding)), torch.tensor([[BOS]]))
    with pytest.raises(ValueError):
        model.decode(memory, torch.tensor([[BOS], [BOS]]))
    with pytest.raises(ValueError):
        model.decode(memory, torch.ones((1, MAX_OUTPUT_TOKENS), dtype=torch.long))
    with pytest.raises(ValueError):
        model.decode(object(), torch.tensor([[BOS]]))


def test_source_counts_prefix_and_memory_must_share_model_device():
    model = tiny()
    with pytest.raises(ValueError, match="device"):
        model.encode(source().to("meta"), torch.tensor([0]))
    with pytest.raises(ValueError, match="device"):
        model.encode(source(), torch.tensor([0], device="meta"))
    memory = model.encode(source(), torch.tensor([0]))
    with pytest.raises(ValueError, match="device"):
        model.decode(memory, torch.tensor([[BOS]], device="meta"))
    with pytest.raises(ValueError, match="device"):
        model.decode(replace(memory, hidden=memory.hidden.to("meta")), torch.tensor([[BOS]]))


def scripted(monkeypatch, outputs):
    model = tiny()
    counters = {"encode": 0, "decode": 0}
    original_encode = model.encode

    def encode(*args):
        counters["encode"] += 1
        assert not model.training and not torch.is_grad_enabled()
        return original_encode(*args)

    def decode(memory, prefix):
        assert type(memory) is EncodedSource
        assert not model.training and not torch.is_grad_enabled()
        token = outputs[min(counters["decode"], len(outputs) - 1)]
        counters["decode"] += 1
        logits = torch.zeros((*prefix.shape, VOCAB_SIZE), device=prefix.device)
        if token == "nonfinite":
            logits[0, -1, EOS] = torch.nan
        else:
            logits[0, -1, token] = 10
        return logits

    monkeypatch.setattr(model, "encode", encode)
    monkeypatch.setattr(model, "decode", decode)
    return model, counters


@pytest.mark.parametrize("training", [False, True])
def test_generate_encodes_once_expands_copy_and_restores_prior_mode(monkeypatch, training):
    model, counters = scripted(monkeypatch, [COPY_BASE, byte("!"), EOS])
    model.train(training)
    prepared = prepare_input('inspect "e\\u0301.py"', 'not "é.py"')
    result = model.generate(prepared, max_new_tokens=8)
    assert result.status == "complete" and result.error is None
    assert result.text == '"e\\u0301.py"!'
    assert result.tokens == (BOS, COPY_BASE, byte("!"), EOS)
    assert counters == {"encode": 1, "decode": 3}
    assert model.training is training and result.elapsed_seconds >= 0


def test_generate_never_forces_eos_at_token_limit_or_silently_returns_source(monkeypatch):
    model, counters = scripted(monkeypatch, [byte("x")])
    result = model.generate(prepare_input("original", ""), max_new_tokens=3)
    assert result.status == "decode_budget_exhausted" and result.text is None
    assert result.tokens == (BOS, byte("x"), byte("x"), byte("x"))
    assert EOS not in result.tokens and result.error
    assert counters == {"encode": 1, "decode": 3}
    assert model.training


def test_generate_invalid_utf8_preserves_raw_output_without_text_or_retry(monkeypatch):
    model, counters = scripted(monkeypatch, [BYTE_BASE + 255, EOS])
    result = model.generate(prepare_input("input", ""), max_new_tokens=5)
    assert result.status == "invalid_output" and result.text is None
    assert result.tokens == (BOS, BYTE_BASE + 255, EOS)
    assert "UTF-8" in result.error and counters["decode"] == 2
    assert model.training


def test_generate_expansion_overflow_is_failure_not_truncation(monkeypatch):
    lexeme = '"' + "x" * 1000 + '"'
    model, counters = scripted(monkeypatch, [COPY_BASE] * 9 + [EOS])
    result = model.generate(prepare_input(lexeme, ""), max_new_tokens=12)
    assert result.status == "invalid_output" and result.text is None
    assert result.tokens == (BOS,) + (COPY_BASE,) * 9 + (EOS,)
    assert "exceeds" in result.error and counters["decode"] == 10


def test_generate_rejects_nonfinite_permitted_logits_without_selecting_a_token(monkeypatch):
    model, counters = scripted(monkeypatch, ["nonfinite"])
    result = model.generate(prepare_input("input", ""), max_new_tokens=2)
    assert result.status == "nonfinite_logits" and result.text is None
    assert result.tokens == (BOS,) and counters == {"encode": 1, "decode": 1}
    assert model.training


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("stage", ["before_encode", "encode", "decode"])
def test_time_budget_is_checked_around_each_expensive_stage(monkeypatch, stage):
    model, counters = scripted(monkeypatch, [byte("x"), EOS])
    clock = Clock()
    if stage == "before_encode":
        times = iter([0.0, 2.0, 2.0])

        def clock():
            return next(times)
    else:
        original = getattr(model, stage)

        def delayed(*args):
            value = original(*args)
            clock.now = 2.0
            return value

        monkeypatch.setattr(model, stage, delayed)
    result = model.generate(prepare_input("input", ""), max_seconds=1, clock=clock)
    assert result.status == "time_budget_exhausted" and result.text is None
    assert result.tokens == (BOS,) and result.elapsed_seconds == 2
    assert counters["encode"] == (0 if stage == "before_encode" else 1)
    assert counters["decode"] == (1 if stage == "decode" else 0)
    assert model.training


def test_timeout_after_copy_expansion_retains_eos_but_does_not_deliver_text(monkeypatch):
    model, _ = scripted(monkeypatch, [EOS])
    clock = Clock()
    from drummer import rewrite_model
    original = rewrite_model.decode_output

    def delayed(*args):
        result = original(*args)
        clock.now = 2.0
        return result

    monkeypatch.setattr(rewrite_model, "decode_output", delayed)
    result = model.generate(prepare_input("input", ""), max_seconds=1, clock=clock)
    assert result.status == "time_budget_exhausted" and result.text is None
    assert result.tokens == (BOS, EOS)


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": 0}, {"max_new_tokens": True}, {"max_new_tokens": 2.0},
    {"max_new_tokens": MAX_OUTPUT_TOKENS}, {"max_seconds": 0},
    {"max_seconds": True}, {"max_seconds": float("nan")}, {"max_seconds": 61},
])
def test_generation_limits_are_strict_before_any_generation(kwargs):
    with pytest.raises(ValueError):
        tiny().generate(prepare_input("input", ""), **kwargs)


def test_unexpected_decoder_exception_still_restores_training_mode(monkeypatch):
    model = tiny()

    def broken(*args):
        raise RuntimeError("synthetic unexpected error")

    monkeypatch.setattr(model, "decode", broken)
    with pytest.raises(RuntimeError, match="synthetic"):
        model.generate(prepare_input("input", ""))
    assert model.training
