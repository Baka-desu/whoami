import argparse

import pytest

from ugv_eval.bool_arg import parse_bool


@pytest.mark.parametrize("text", ["true", "True", "1", "on", "yes"])
def test_parse_bool_true(text: str) -> None:
    assert parse_bool(text) is True


@pytest.mark.parametrize("text", ["false", "False", "0", "off", "no"])
def test_parse_bool_false(text: str) -> None:
    assert parse_bool(text) is False


def test_parse_bool_rejects_garbage() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("maybe")
