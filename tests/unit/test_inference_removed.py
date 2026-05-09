"""Regression tests pinning the removal of snowgan.inference.

The transfer-learning pipeline that used to live in snowgan.inference
moved to the AvAI library. The unblockers doc treats this as a hard
contract: importing the module must fail loud (so any forgotten caller
surfaces immediately), and the CLI must redirect users to AvAI rather
than silently accepting --mode infer.
"""
from __future__ import annotations

import importlib

import pytest


def test_inference_module_no_longer_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("snowgan.inference")


def test_main_redirects_infer_mode_to_avai(monkeypatch, capsys):
    from snowgan import main as main_module

    monkeypatch.setattr(
        "sys.argv", ["snowgan", "--mode", "infer", "--save_dir", "/tmp/snowgan-test"]
    )

    with pytest.raises(SystemExit) as excinfo:
        main_module.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "AvAI" in captured.err
    assert "github.com/dennys246/AvAI" in captured.err


def test_infer_samples_flag_removed():
    from snowgan.utils import parse_args
    import sys

    saved_argv = sys.argv
    try:
        sys.argv = [
            "snowgan",
            "--mode",
            "train",
            "--infer_samples",
            "100",
        ]
        with pytest.raises(SystemExit):
            parse_args()
    finally:
        sys.argv = saved_argv
