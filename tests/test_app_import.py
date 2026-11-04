"""The desk app imports and exposes a main function. Widget behavior is exercised by hand."""

from __future__ import annotations

import importlib.util

from credit_risk.config import PROJECT_ROOT


def test_streamlit_app_imports():
    path = PROJECT_ROOT / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("credit_desk_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
