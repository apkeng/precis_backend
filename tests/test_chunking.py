from app.chunking import chunk_pages


def test_short_page_is_a_single_chunk():
    chunks = chunk_pages(["hello world"], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_long_page_is_split_with_overlap():
    text = "abcdefghij" * 30  # 300 chars
    chunks = chunk_pages([text], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    # chunk_index is assigned sequentially, starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_numbers_tracked_across_multiple_pages():
    chunks = chunk_pages(["page one", "page two", "page three"], chunk_size=100, chunk_overlap=10)
    assert [c.page_number for c in chunks] == [1, 2, 3]


def test_empty_pages_produce_no_chunks():
    chunks = chunk_pages(["", "   ", "real content"], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "real content"
    assert chunks[0].page_number == 3


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages(["text"], chunk_size=50, chunk_overlap=50)
