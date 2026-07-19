from pathlib import Path


GUIDE_PATH = (
    Path(__file__).parents[1]
    / "ml_logger"
    / "docs"
    / "ml_logger_guide.html"
)


def test_visual_guide_is_a_plain_standalone_document():
    html = GUIDE_PATH.read_text(encoding="utf-8")

    assert html.startswith("<!doctype html>")
    assert "<iframe" not in html
    assert "srcdoc=" not in html
    assert "https://" not in html


def test_visual_guide_tabs_have_matching_panels():
    html = GUIDE_PATH.read_text(encoding="utf-8")
    panel_names = (
        "architettura",
        "quickstart",
        "dashboard",
        "configurazione",
        "funzionalita",
        "adapter",
    )

    for panel_name in panel_names:
        assert f'data-panel="{panel_name}"' in html
        assert f'id="{panel_name}" role="tabpanel"' in html
