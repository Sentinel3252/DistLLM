import importlib
import json
import sys

import pytest

reproduction = importlib.import_module("distllm.reproduce")


@pytest.mark.parametrize("custom", [False, True])
def test_reproduction_output_path(tmp_path, monkeypatch, capsys, custom):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / ("含 空格/nested/result.json" if custom else "results/reproduction.json")
    args = ["reproduce", "--output", str(destination)] if custom else ["reproduce"]
    monkeypatch.setattr(sys, "argv", args)
    result = {"reference_loss": 1.0, "fixed_gradient_relative_l2": 0.0}
    monkeypatch.setattr(reproduction, "reproduce", lambda: result)
    reproduction.main()
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert json.loads(capsys.readouterr().out) == result


def test_reproduction_help_does_not_start_computation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reproduce", "--help"])
    monkeypatch.setattr(reproduction, "reproduce", lambda: pytest.fail("reproduction started"))
    with pytest.raises(SystemExit) as error:
        reproduction.main()
    assert error.value.code == 0
    assert "--output" in capsys.readouterr().out
