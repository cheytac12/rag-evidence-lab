import pytest

from src import generator
from src.schemas import RetrievedPassage


def test_missing_openai_config_blocks_before_sdk_call(monkeypatch):
    monkeypatch.setattr(generator, "openai_configured", lambda *args, **kwargs: False)

    passage = RetrievedPassage(1, "p1", "d", "Doc", "file.pdf", 1, 1, "evidence", 1.0, "bm25")
    with pytest.raises(generator.MissingOpenAIConfiguration):
        generator.generate_answer("Question?", [passage])
