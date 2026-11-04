"""The Community Cloud entry point imports, and the repo can score offline."""

from __future__ import annotations

import importlib.util

from credit_risk.config import PROJECT_ROOT


def test_streamlit_app_imports():
    path = PROJECT_ROOT / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("credit_desk_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module.cached_artifact)


def test_entrypoint_does_not_download_data():
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "fetch_openml" not in source
    assert "load_german_credit" not in source
    assert "load_artifact" in source
    assert "explain_applicant" in source
    assert "st.tabs" in source
    assert "st.file_uploader" in source
    assert "st.slider" in source
    assert "score_batch" in source


def test_requirements_are_pinned():
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "==" in stripped
        assert stripped.count("==") == 1
        name, version = stripped.split("==")
        assert name and version
        assert not name.startswith("-")
        packages.append(name)
    for pin in ("numpy==2.5.4", "pandas==3.1.0", "scikit-learn==1.9.1", "streamlit==1.66.1"):
        assert pin in "\n".join(lines)
    assert "streamlit" in packages


def test_streamlit_config_is_offline_and_at_the_repo_root():
    path = PROJECT_ROOT / ".streamlit" / "config.toml"
    text = path.read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in text
    assert "headless = true" in text
    assert "[theme]" in text
    assert 'primaryColor = "#0F6E56"' in text
    assert "showWidgetBorder = true" in text
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
