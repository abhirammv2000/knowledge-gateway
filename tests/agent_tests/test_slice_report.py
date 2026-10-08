"""Slice analysis of a saved run. Pure statistics over rows, no model and no CUAD file needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import slice_report as sr  # noqa: E402


def row(contract, category, correct, answerable=True):
    return {"id": f"{contract}__{category}", "category": category, "correct": correct, "answerable": answerable}


def test_wilson_matches_known_values_and_handles_the_edges():
    low, high = sr.wilson(8, 10)
    assert low == pytest.approx(0.490, abs=0.005) and high == pytest.approx(0.943, abs=0.005)
    assert sr.wilson(0, 0) == (0.0, 1.0)
    assert sr.wilson(0, 5)[0] == 0.0 and sr.wilson(5, 5)[1] == 1.0


@pytest.mark.parametrize("title, expected", [
    ("TubeMediaCorp_20060310_8-K_EX-10.1_513921_EX-10.1_Affiliate Agreement", "affiliate agreement"),
    ("ACME_20010101_10-K_EX-10_1_DISTRIBUTOR AGREEMENT", "distributor agreement"),
    ("NoUnderscoresHere", "nounderscoreshere"),
    ("PLUM_20120101_10-K_2012-EX-10.6-TRANSPORTATION CONTRACT", "transportation contract"),
    ("X_1999-EX-10.22-MAINTENANCE AGREEMENT", "maintenance agreement"),
    ("X_EX-10.1_DISTRIBUTOR AGREEMENT2", "distributor agreement"),
])
def test_the_contract_type_is_read_from_the_end_of_the_title(title, expected):
    assert sr.contract_type(title) == expected


def test_a_row_id_gives_back_its_contract():
    assert sr.contract_of(row("My_Contract_Agreement", "Governing Law", True)) == "My_Contract_Agreement"


def test_contracts_are_split_into_thirds_by_length():
    bands = sr.length_bands({"a": 100, "b": 200, "c": 300, "d": 400, "e": 500, "f": 600})

    assert bands == {"a": "short", "b": "short", "c": "medium", "d": "medium", "e": "long", "f": "long"}


def test_slices_group_header_facts_apart_from_clause_terms_and_marked_from_unmarked():
    rows = [row("x_A", "Parties", True), row("x_A", "Governing Law", True), row("x_A", "Non-Compete", False, answerable=False)]

    got = sr.slices(rows)

    assert len(got["clause group"]["header facts"]) == 1 and len(got["clause group"]["clause terms"]) == 2
    assert len(got["clause present"]["clause marked"]) == 2 and len(got["clause present"]["no clause marked"]) == 1


def test_a_clearly_worse_slice_is_flagged():
    rows = [row("x_A", "Parties", False) for _ in range(12)] + [row("x_B", "Governing Law", True) for _ in range(40)]

    entries = {(e["dimension"], e["slice"]): e for e in sr.report(rows)}

    assert entries[("clause group", "header facts")]["clearly_worse"] is True
    assert entries[("clause group", "clause terms")]["clearly_worse"] is False


def test_a_tiny_slice_is_never_flagged_however_bad_it_looks():
    rows = [row("x_A", "Parties", False), row("x_A", "Document Name", False)] + [row("x_B", "Governing Law", True) for _ in range(40)]

    entries = {(e["dimension"], e["slice"]): e for e in sr.report(rows)}

    assert entries[("clause group", "header facts")]["n"] == 2 and entries[("clause group", "header facts")]["clearly_worse"] is False


def test_a_slice_whose_interval_reaches_the_rest_is_not_flagged():
    # 3 of 6 is 50% against 75% elsewhere, but six questions cannot rule out luck
    rows = [row("x_A", "Parties", i < 3) for i in range(6)] + [row("x_B", "Governing Law", i < 30) for i in range(40)]

    entries = {(e["dimension"], e["slice"]): e for e in sr.report(rows)}

    assert entries[("clause group", "header facts")]["accuracy"] == 0.5
    assert entries[("clause group", "header facts")]["clearly_worse"] is False


def test_contract_types_with_fewer_than_three_questions_are_left_out():
    rows = [row("a_b_Rare Deal", "Governing Law", True)] + [row("a_b_Common Deal", "Governing Law", True) for _ in range(4)]

    types = sr.slices(rows)["contract type"]

    assert set(types) == {"common deal"}


def test_length_slices_appear_only_when_word_counts_are_known():
    rows = [row("x_A", "Parties", True), row("x_B", "Parties", True), row("x_C", "Parties", True)]

    assert sr.slices(rows)["contract length"] == {}
    assert set(sr.slices(rows, {"x_A": 10, "x_B": 20, "x_C": 30})["contract length"]) == {"short", "medium", "long"}


def test_many_slices_make_the_flag_stricter_so_a_lucky_small_one_is_not_reported():
    # twenty contract types of five questions each, one of them at 40% while the rest are near 90%
    rows = []
    for k in range(20):
        right = 2 if k == 0 else 4
        rows += [row(f"c{k}_{chr(97 + k) * 4} deal", "Governing Law", i < right) for i in range(5)]

    entries = [e for e in sr.report(rows) if e["dimension"] == "contract type"]

    assert len(entries) == 20 and not any(e["clearly_worse"] for e in entries)
