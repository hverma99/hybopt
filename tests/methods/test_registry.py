import pytest

from hybopt.methods import METHODS, get_method
from hybopt.methods.relu import bigm_ia


def test_registered_methods():
    assert METHODS == {"relu.bigm_ia": bigm_ia}


def test_get_method():
    assert get_method("relu.bigm_ia", "relu") is bigm_ia


@pytest.mark.parametrize("key", ["relu.bigm", "bigm_ia", "", "RELU.BIGM_IA"])
def test_unknown_method(key):
    with pytest.raises(KeyError, match=r"unknown method .*; available: \['relu.bigm_ia'\]"):
        get_method(key, "relu")


def test_method_for_another_activation():
    with pytest.raises(
        ValueError, match="method 'relu.bigm_ia' is for relu networks, but the network uses tanh"
    ):
        get_method("relu.bigm_ia", "tanh")
