import sys

import pytest
import torch.distributed as dist

from distllm import launch_cpu


@pytest.mark.parametrize("arguments,message", [
    (["--nproc-per-node", "0"], "positive"),
    (["--backend", "nccl"], "gloo"),
    (["--mode", "reference"], "reference"),
    (["--batch", "1"], "batch"),
    (["--mode", "pp", "--layers", "3"], "layers"),
    (["--mode", "tp", "--nproc-per-node", "3"], "hidden width"),
])
def test_invalid_launch_fails_before_creating_store(arguments, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["launch", *arguments])
    monkeypatch.setattr(dist, "TCPStore", lambda *a, **kw: pytest.fail("store started"))
    with pytest.raises(SystemExit) as error:
        launch_cpu.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
