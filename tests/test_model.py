import pytest
import torch

from distllm.model import Config, Transformer


@pytest.mark.parametrize("field", ["vocab", "width", "heads", "layers", "expansion"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_rejects_invalid_dimensions(field, value):
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible by heads"):
        Config(width=10, heads=3)


def test_nondefault_model_dimensions():
    model = Transformer(Config(vocab=11, width=12, heads=3, layers=1, expansion=2))
    assert model(torch.randint(11, (2, 5))).shape == (2, 5, 11)
