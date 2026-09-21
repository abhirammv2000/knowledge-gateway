import pytest

from gateway.chunking import fixed_window_chunks, structure_aware_chunks


def covered_chars(text, chunks):
    mask = [False] * len(text)
    for c in chunks:
        for i in range(c.start, c.end):
            mask[i] = True
    return mask


def assert_every_nonspace_char_is_covered(text, chunks):
    mask = covered_chars(text, chunks)
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and not mask[i]]
    assert missing == []


def body(n, prefix="word"):
    return " ".join(f"{prefix}{i}" for i in range(n))


# ---- fixed window ----

def test_fixed_window_offsets_match_text_and_overlap_is_applied():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = fixed_window_chunks(text, size=4, overlap=1)

    for c in chunks:
        assert text[c.start : c.end] == c.text
    assert chunks[0].text == "w0 w1 w2 w3"
    assert chunks[1].text.split()[0] == "w3"  # one word of overlap
    assert_every_nonspace_char_is_covered(text, chunks)


def test_fixed_window_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        fixed_window_chunks("a b c", size=3, overlap=3)


def test_fixed_window_on_empty_text_is_empty():
    assert fixed_window_chunks("") == []


# ---- structure aware ----

def test_each_clause_becomes_its_own_chunk_headed_by_its_heading():
    text = f"1. Payment. {body(30)}\n\n2. Termination. {body(30, 'x')}"
    chunks = structure_aware_chunks(text, max_words=100, min_words=10)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("1. Payment.")
    assert chunks[1].text.startswith("2. Termination.")
    assert "Termination" not in chunks[0].text


def test_heading_stays_with_the_paragraphs_that_follow_it():
    text = f"GOVERNING LAW\n\n{body(30)}\n\n{body(20, 'y')}"
    chunks = structure_aware_chunks(text, max_words=200, min_words=10)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("GOVERNING LAW")
    assert "y19" in chunks[0].text


def test_tiny_section_is_merged_into_the_next_instead_of_standing_alone():
    text = f"ARTICLE I\n\n1. Definitions. {body(40)}"
    chunks = structure_aware_chunks(text, max_words=200, min_words=15)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("ARTICLE I")


def test_oversized_section_is_split_under_the_limit_and_later_pieces_keep_the_heading():
    sentences = " ".join(f"Sentence number {i} is here." for i in range(60))
    text = f"5. Indemnification. {sentences}"
    chunks = structure_aware_chunks(text, max_words=50, min_words=10)

    assert len(chunks) > 1
    for c in chunks:
        assert len(text[c.start : c.end].split()) <= 50
    for c in chunks[1:]:
        assert c.text.startswith("5. Indemnification.")
    assert_every_nonspace_char_is_covered(text, chunks)


def test_a_single_sentence_longer_than_the_limit_is_still_split():
    text = "1. Scope. " + body(120)
    chunks = structure_aware_chunks(text, max_words=40, min_words=5)

    assert len(chunks) >= 3
    for c in chunks:
        assert len(text[c.start : c.end].split()) <= 40
    assert_every_nonspace_char_is_covered(text, chunks)


def test_no_text_is_lost_across_a_multi_clause_document():
    text = (
        "MASTER SERVICES AGREEMENT\n\n"
        f"1. Services. {body(60)}\n\n"
        f"(a) sub item {body(10, 'a')}\n\n"
        f"2. Fees. {body(45, 'f')}\n\n"
        "IN WITNESS WHEREOF the parties have executed this Agreement."
    )
    chunks = structure_aware_chunks(text, max_words=80, min_words=12)

    assert_every_nonspace_char_is_covered(text, chunks)
    for c in chunks:
        assert c.start < c.end


def test_structure_aware_on_empty_text_is_empty():
    assert structure_aware_chunks("") == []
    assert structure_aware_chunks("   \n\n  ") == []
