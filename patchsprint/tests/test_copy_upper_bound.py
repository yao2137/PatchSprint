"""Toy-trace tests for the replay and proposers in analysis/copy_upper_bound.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from copy_upper_bound import (  # noqa: E402
    NgramProposer,
    NoSpecProposer,
    AlignOracleProposer,
    ContextOracleProposer,
    HybridProposer,
    PointerProposer,
    matching_blocks,
    replay,
)

SRC = list(range(100, 140))  # 40 distinct tokens


def oracle(out, src=SRC):
    return AlignOracleProposer(out, matching_blocks(src, out))


def test_no_spec_takes_one_step_per_token():
    out = SRC[:10]
    assert replay(out, NoSpecProposer(), k=8).steps == 10


def test_identical_file_oracle_and_pointer_reach_k_plus_one():
    k = 7
    expected = math.ceil(len(SRC) / (k + 1))
    assert replay(SRC, oracle(SRC), k).steps == expected
    assert replay(SRC, PointerProposer(SRC), k).steps == expected


def test_insertion_pointer_resyncs_to_same_position():
    inserted = [900, 901, 902, 903, 904]
    out = SRC[:20] + inserted + SRC[20:]
    st = replay(out, PointerProposer(SRC), k=8)
    # Must beat plain decoding by a wide margin and never beat the oracle.
    assert st.steps < len(out) / 3
    assert st.steps >= replay(out, oracle(out), k=8).steps


def test_deletion_pointer_jumps_forward():
    out = SRC[:15] + SRC[25:]
    st = replay(out, PointerProposer(SRC), k=8)
    assert st.steps < len(out) / 3


def test_replay_never_accepts_wrong_tokens():
    # Whatever the proposer drafts, the replay must reproduce the output exactly,
    # so accepted + bonus tokens over all steps equal the output length.
    out = SRC[:10] + [900, 901] + SRC[12:30]
    for prop in (PointerProposer(SRC), NgramProposer(SRC), oracle(out)):
        st = replay(out, prop, k=6)
        assert st.accepted + st.steps >= len(out)
        assert st.accepted <= len(out)


def test_ngram_prefers_earliest_match_on_repeated_code():
    # Source contains the pattern A B C twice with different continuations;
    # the stateless n-gram proposer picks the earliest one.
    a, b, c = 1, 2, 3
    src = [a, b, c, 10, 11, 50, a, b, c, 20, 21]
    prop = NgramProposer(src, min_n=2, max_n=3)
    draft = prop.propose([a, b, c], k=2)
    assert draft == [10, 11]


def test_context_oracle_bounds_every_copy_proposer():
    # New code that reuses a fragment from elsewhere in the source (non-monotonic).
    out = SRC[:10] + SRC[30:36] + SRC[10:25]
    best = replay(out, ContextOracleProposer(SRC, out), k=8).steps
    for prop in (PointerProposer(SRC), NgramProposer(SRC), HybridProposer(SRC), oracle(out)):
        assert replay(out, prop, k=8).steps >= best
