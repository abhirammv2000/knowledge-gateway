"""Two chunkers over the same text, to compare them. Every chunk keeps character offsets into the original text,
which is how chunking_eval.py scores it against the marked clause spans."""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\S+")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"(?<=[.;])\s+")

# numbered clauses ("2.", "2.1 Payment") and ARTICLE/SECTION headings, picked from the data: numbered
# headings are in 447 of 510 contracts, ARTICLE/SECTION in 112, all-caps lines in 310
_NUMBERED_OR_ARTICLE = re.compile(
    r"^\s*(?:(?:ARTICLE|Article|SECTION|Section)\s+[\dIVXivx]+\b|\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z])"
)
_ALL_CAPS = re.compile(r"^\s*[A-Z][A-Z0-9 \-,&'\.]{5,}\s*$")


@dataclass
class Chunk:
    start: int  # offsets into the original text
    end: int
    text: str  # what gets indexed; may carry a section-heading prefix


def _n_words(s: str) -> int:
    return len(s.split())


def fixed_window_chunks(text: str, size: int = 250, overlap: int = 50) -> list[Chunk]:
    """Baseline: sliding window of `size` words, `overlap` words shared."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        j = min(i + size, len(words))
        s, e = words[i][0], words[j - 1][1]
        chunks.append(Chunk(s, e, text[s:e]))
        if j == len(words):
            break
        i += size - overlap
    return chunks


def _blocks(text: str) -> list[tuple[int, int]]:
    """Paragraph spans (split on blank lines), trimmed of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in list(_BLANK_LINE.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        raw = text[cursor:end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            spans.append((cursor + lead, cursor + lead + len(stripped)))
        cursor = m.end() if m else len(text)
    return spans


def _is_heading(block: str) -> bool:
    if _NUMBERED_OR_ARTICLE.match(block):
        return True
    return _n_words(block) <= 12 and "\n" not in block.strip() and bool(_ALL_CAPS.match(block))


def _split_span(text: str, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    """Cut a too-long span at paragraph, then sentence, then word boundaries."""
    if _n_words(text[start:end]) <= max_words:
        return [(start, end)]

    units: list[tuple[int, int]] = []
    for bs, be in _blocks(text[start:end]):
        units.append((start + bs, start + be))

    expanded: list[tuple[int, int]] = []
    for us, ue in units:
        if _n_words(text[us:ue]) <= max_words:
            expanded.append((us, ue))
            continue
        # sentence level
        pos = us
        pieces: list[tuple[int, int]] = []
        for m in _SENTENCE_END.finditer(text[us:ue]):
            pieces.append((pos, us + m.start()))
            pos = us + m.end()
        pieces.append((pos, ue))
        for ps, pe in pieces:
            if _n_words(text[ps:pe]) <= max_words:
                expanded.append((ps, pe))
            else:  # a single sentence over the limit: hard word split
                words = [(ps + m.start(), ps + m.end()) for m in _WORD.finditer(text[ps:pe])]
                for i in range(0, len(words), max_words):
                    expanded.append((words[i][0], words[min(i + max_words, len(words)) - 1][1]))

    # greedily pack the units back up to max_words
    packed: list[tuple[int, int]] = []
    cur_s, cur_e, cur_w = expanded[0][0], expanded[0][1], _n_words(text[expanded[0][0] : expanded[0][1]])
    for us, ue in expanded[1:]:
        w = _n_words(text[us:ue])
        if cur_w + w <= max_words:
            cur_e, cur_w = ue, cur_w + w
        else:
            packed.append((cur_s, cur_e))
            cur_s, cur_e, cur_w = us, ue, w
    packed.append((cur_s, cur_e))
    return packed


def structure_aware_chunks(text: str, max_words: int = 250, min_words: int = 40) -> list[Chunk]:
    """One chunk per clause where it can.

    A new section starts at each heading (numbered clause, ARTICLE/SECTION, short all-caps line). Sections under
    min_words are merged into the next one so a bare heading isn't indexed alone. Sections over max_words are split
    at paragraphs, then sentences, then words, and later pieces get the heading put back in front.
    """
    blocks = _blocks(text)
    if not blocks:
        return []

    sections: list[list[tuple[int, int]]] = []
    for bs, be in blocks:
        if _is_heading(text[bs:be]) or not sections:
            sections.append([(bs, be)])
        else:
            sections[-1].append((bs, be))

    spans = [(sec[0][0], sec[-1][1]) for sec in sections]

    merged: list[tuple[int, int]] = []
    carry_start: int | None = None
    for s, e in spans:
        start = carry_start if carry_start is not None else s
        if _n_words(text[start:e]) < min_words:
            carry_start = start
            continue
        merged.append((start, e))
        carry_start = None
    if carry_start is not None:
        if merged:
            merged[-1] = (merged[-1][0], spans[-1][1])
        else:
            merged.append((carry_start, spans[-1][1]))

    chunks: list[Chunk] = []
    for s, e in merged:
        heading = " ".join(text[s:e].split()[:12])
        for i, (ps, pe) in enumerate(_split_span(text, s, e, max_words)):
            body = text[ps:pe]
            chunks.append(Chunk(ps, pe, body if i == 0 else f"{heading}\n{body}"))
    return chunks
