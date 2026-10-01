from app.utils import chunk_text, make_metadata
def test_chunking_and_metadata():
    text = " ".join([f"word{i}" for i in range(1200)])
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 0
    meta = make_metadata("t", "s", 0)
    assert "chunk_id" in meta
