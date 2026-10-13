import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.chunking import (
    chunks_from_parse_run, estimate_tokens, num_pages_in_run, re_chunk_text,
)

FAKE_RUN = {
    "id": "pr_test",
    "status": "PROCESSED",
    "output": {
        "chunks": [
            {
                "content": "# Refund policy\n\nRefunds within 30 days.",
                "blocks": [
                    {"type": "text", "content": "Refund policy", "page": 1},
                    {"type": "text", "content": "Refunds within 30 days.", "page": 1},
                ],
            },
            {
                "content": "## Shipping\n\nShips in 2 days.",
                "blocks": [{"type": "table", "content": "| a | b |", "pageNumber": 2}],
            },
            {"content": "   ", "blocks": []},  # empty chunk is dropped
        ]
    },
}


def test_chunks_from_parse_run():
    chunks = chunks_from_parse_run(FAKE_RUN, "dochash1")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "dochash1:00000"
    assert chunks[0].ordinal == 0
    assert "Refund policy" in chunks[0].content
    assert chunks[0].token_estimate > 0
    import json
    assert json.loads(chunks[0].pages_json) == [1]
    assert json.loads(chunks[1].pages_json) == [2]


def test_num_pages_in_run():
    assert num_pages_in_run(FAKE_RUN) == 2
    assert num_pages_in_run({}) == 0


def test_estimate_tokens():
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1


def test_re_chunk_text():
    text = " ".join(f"w{i}" for i in range(100))
    parts = re_chunk_text(text, max_tokens=40, overlap_tokens=8)
    assert len(parts) > 1
    assert parts[0].startswith("w0")
    # overlap: second window starts before the first one ends
    assert "w29" in parts[1] or "w28" in parts[1]
    assert re_chunk_text("") == []
