import pandas as pd

from openplaque.multivessel_research_summary_v1 import (
    BASELINE,
    STATUS,
    synthetic_self_test,
)


def test_constants_and_self_test():
    assert BASELINE == "0593b453959f5a353d644267fbeef24b514ef4d7"
    assert STATUS == "OPENPLAQUE_MULTIVESSEL_RESEARCH_SUMMARY_COMPLETE"
    assert synthetic_self_test()["ok"] is True


def test_status_semantics_are_research_only():
    assert "RESEARCH" in STATUS
