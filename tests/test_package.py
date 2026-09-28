import distllm


def test_version_is_exposed():
    assert distllm.__version__ == "0.1.0"
