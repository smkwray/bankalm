from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8").lower()


def test_public_docs_do_not_claim_calibrated_probability_without_calibration() -> None:
    public_text = "\n".join(
        [
            _read("README.md"),
            _read("site", "index.html"),
            _read("site", "bank.html"),
            _read("site", "league.html"),
        ]
    )
    assert "calibrated probability" not in public_text
    assert "calibrated p(severe outflow)" not in public_text


def test_public_docs_do_not_repeat_old_failure_backtest_design() -> None:
    public_text = "\n".join([_read("README.md"), _read("site", "index.html")])
    assert "last pre-failure quarter vs last quarter for survivors" not in public_text


def test_public_docs_describe_proxy_outputs_cautiously() -> None:
    readme = _read("README.md")
    site_index = _read("site", "index.html")
    assert "scenario proxies" in readme
    assert "structural public-data proxies" in readme
    assert "scenario proxies" in site_index
    assert "deposit competition now participates in the same quarter-aligned failure backtest" in readme
    assert "transparent public-data pressure screen for exploratory comparison" in readme
    assert "transparent public-data pressure screen rather than a complete funding diagnostic" in readme
