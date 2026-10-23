from pathlib import Path

from streamlit.testing.v1 import AppTest
from src.generator import TRANSMISSION_NOTICE


def test_all_seven_pages_render_without_exception():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    assert not app.exception
    for page in ["Overview", "Corpus", "Ask", "Experiments", "Claims", "Retrieval debugger", "Exports"]:
        app.radio(key="navigation").set_value(page).run()
        assert not app.exception, f"Streamlit page failed: {page}"


def test_ask_page_shows_notice_and_missing_openai_state(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
    app.selectbox(key="active_data_label").set_value("DEMO").run()
    app.radio(key="navigation").set_value("Ask").run()
    assert any(TRANSMISSION_NOTICE in warning.value for warning in app.warning)
    app.selectbox(key="ask_method").set_value("BM25").run()
    app.text_area(key="ask_question").set_value("When is a record current?").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Missing OpenAI configuration" in item.value for item in app.info)
