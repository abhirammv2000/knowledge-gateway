"""Has the agent started behaving differently? Compares the last few days of requests with the days before.

Nobody labels live questions, so accuracy cannot be watched directly. What can be watched is how the
agent behaves: how often it answers, how often it finds something, how often a provider fails or a
fallback is used, and how long and how much a question takes. A model update at the provider, a changed
prompt, a new batch of contracts or a different kind of user all show up as a shift in these.

A shift is flagged only when both windows have enough requests. Rates use a two-proportion z-test with a
strict cutoff, because several metrics are checked at once and a loose cutoff would flag noise. Sizes use
a ratio of medians. A flag says "look at this", not "this is broken".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

MIN_REQUESTS = 30
Z_CUTOFF = 3.0
RATIO_CUTOFF = 1.5

RATES = {
    "answered": lambda r: r["status"] == "answered",
    "provider failures": lambda r: r["status"] in ("error", "timeout"),
    "fallback used": lambda r: bool(r["fallback"]),
}
SIZES = {
    "seconds": lambda r: r["seconds"],
    "cost_usd": lambda r: r["cost_usd"],
    "tool calls": lambda r: r["tool_calls"],
}


@dataclass(frozen=True)
class Finding:
    metric: str
    baseline: float
    recent: float
    flagged: bool
    detail: str


def _z(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    pooled = (successes_a + successes_b) / (n_a + n_b)
    spread = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if spread == 0:
        return 0.0
    return (successes_b / n_b - successes_a / n_a) / spread


def compare(baseline: list[dict], recent: list[dict]) -> list[Finding]:
    """Findings for every metric. With too few requests in either window nothing is flagged."""
    enough = len(baseline) >= MIN_REQUESTS and len(recent) >= MIN_REQUESTS
    findings: list[Finding] = []

    def add_rate(name: str, pick, rows_a: list[dict], rows_b: list[dict]) -> None:
        if not rows_a or not rows_b:
            return
        a, b = sum(map(pick, rows_a)), sum(map(pick, rows_b))
        z = _z(a, len(rows_a), b, len(rows_b))
        ok = len(rows_a) >= MIN_REQUESTS and len(rows_b) >= MIN_REQUESTS
        flagged = ok and abs(z) >= Z_CUTOFF
        findings.append(Finding(name, a / len(rows_a), b / len(rows_b), flagged, f"z = {z:+.1f}"))

    for name, pick in RATES.items():
        add_rate(name, lambda r, pick=pick: int(pick(r)), baseline, recent)
    # of the answered questions, how many found something. Refusals and failures say nothing about this.
    add_rate("found something", lambda r: int(bool(r.get("found"))),
             [r for r in baseline if r["status"] == "answered"], [r for r in recent if r["status"] == "answered"])

    for name, pick in SIZES.items():
        if not baseline or not recent:
            continue
        before, after = median(map(pick, baseline)), median(map(pick, recent))
        ratio = after / before if before else (math.inf if after else 1.0)
        flagged = enough and (ratio >= RATIO_CUTOFF or ratio <= 1 / RATIO_CUTOFF)
        findings.append(Finding(f"median {name}", before, after, flagged, f"x{ratio:.2f}"))
    return findings
