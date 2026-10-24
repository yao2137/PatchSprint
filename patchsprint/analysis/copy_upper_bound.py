"""Copyable-output upper bound and trace-driven proposer simulation for code edits.

For every CanItEdit problem we treat the reference solution (`after`) as the model
output and the original file (`before`) as the input, then measure:

1. How much of the output is copied verbatim from the input (token and char level).
2. The decode-step speedup an *oracle* proposer would reach with draft budget k.
3. The decode-step speedup of two concrete proposers replayed on the same trace:
   - "ngram":   a vLLM-style stateless n-gram lookup (longest suffix match,
                ties broken by the earliest position, searches input + output).
   - "pointer": the PatchSprint alignment-pointer proposer (COPY / EDIT / RESYNC).

Assumptions (state them whenever the numbers are quoted):
- Teacher forcing: the model is assumed to emit exactly the reference solution.
- Speedup is measured in decode *steps* (forward passes), i.e. verification of k
  tokens is assumed to cost the same as one decode step. This holds only in the
  memory-bound regime (small batch); wall-clock speedup will be lower.
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Speculative decoding replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayStats:
    steps: int = 0          # forward passes of the target model
    proposed: int = 0       # draft tokens sent for verification
    accepted: int = 0       # draft tokens accepted

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def replay(output: list[int], proposer, k: int) -> ReplayStats:
    """Greedy speculative decoding replay against a fixed output sequence.

    Each step the proposer drafts up to k tokens; the longest prefix that matches
    the true output is accepted and the target model contributes one bonus token,
    so a step always produces accepted + 1 tokens (fewer at the end of the output).
    """
    stats = ReplayStats()
    i, n = 0, len(output)
    while i < n:
        draft = proposer.propose(output[:i], k)
        a = 0
        while a < len(draft) and i + a < n and draft[a] == output[i + a]:
            a += 1
        produced = min(a + 1, n - i)
        stats.steps += 1
        stats.proposed += len(draft)
        stats.accepted += a
        proposer.update(output[: i + produced], accepted=a, drafted=len(draft))
        i += produced
    return stats


class NoSpecProposer:
    def propose(self, generated: list[int], k: int) -> list[int]:
        return []

    def update(self, generated: list[int], accepted: int, drafted: int) -> None:
        pass


class AlignOracleProposer:
    """Knows the exact monotonic (diff) alignment between input and output.

    It drafts the remainder of the current copied span (capped at k) and nothing
    inside newly written regions. This bounds proposers that only follow the diff
    alignment; it is NOT a bound for proposers that also copy non-monotonically
    or from their own output (see ContextOracleProposer).
    """

    def __init__(self, output: list[int], blocks: list[tuple[int, int, int]]):
        # copy_left[j] = number of consecutive copied tokens starting at output[j]
        self.copy_left = np.zeros(len(output) + 1, dtype=np.int64)
        for _, b, size in blocks:
            for t in range(size):
                self.copy_left[b + t] = size - t
        self.output = output

    def propose(self, generated: list[int], k: int) -> list[int]:
        i = len(generated)
        r = int(self.copy_left[i]) if i < len(self.output) else 0
        return self.output[i : i + min(k, r)]

    def update(self, generated: list[int], accepted: int, drafted: int) -> None:
        pass


class ContextOracleProposer:
    """Upper bound for every proposer that drafts by copying from its context.

    At each step it drafts the longest prefix of the true continuation that occurs
    anywhere in (source + generated output). Greedy is optimal here: the longest
    match at i+1 is at least the longest match at i minus one, so the furthest
    reachable position is non-decreasing in i.
    """

    def __init__(self, source: list[int], output: list[int]):
        self.ctx = list(source)
        self.n_source = len(source)
        self.output = output
        self.positions: dict[int, list[int]] = defaultdict(list)
        for p, t in enumerate(self.ctx):
            self.positions[t].append(p)

    def propose(self, generated: list[int], k: int) -> list[int]:
        for t in generated[len(self.ctx) - self.n_source :]:
            self.positions[t].append(len(self.ctx))
            self.ctx.append(t)
        i, L = len(generated), len(self.ctx)
        target = self.output[i : i + k]
        if not target:
            return []
        best = 0
        for p in self.positions.get(target[0], ()):
            m = 0
            while m < len(target) and p + m < L and self.ctx[p + m] == target[m]:
                m += 1
            if m > best:
                best = m
                if best == len(target):
                    break
        return target[:best]

    def update(self, generated: list[int], accepted: int, drafted: int) -> None:
        pass


class NgramProposer:
    """vLLM-style prompt lookup: stateless, longest suffix match, earliest wins.

    The context it searches is input + generated output, like vLLM's n-gram
    proposer (which also sees the instruction text; we omit it here, which can
    only help this baseline).
    """

    def __init__(self, source: list[int], min_n: int = 2, max_n: int = 5):
        self.ctx = list(source)
        self.n_source = len(source)
        self.min_n, self.max_n = min_n, max_n
        # Incremental cache of the earliest end position (exclusive) of every n-gram.
        # It only speeds up the search; the proposer's decisions stay stateless.
        self.earliest: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in range(min_n, max_n + 1)
        }
        self.indexed_upto = 0

    def _extend_index(self) -> None:
        for e in range(self.indexed_upto + 1, len(self.ctx) + 1):
            for n, table in self.earliest.items():
                if e >= n:
                    table.setdefault(tuple(self.ctx[e - n : e]), e)
        self.indexed_upto = len(self.ctx)

    def propose(self, generated: list[int], k: int) -> list[int]:
        self.ctx.extend(generated[len(self.ctx) - self.n_source :])
        self._extend_index()
        L = len(self.ctx)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            e = self.earliest[n].get(tuple(self.ctx[L - n :]))
            if e is not None and e < L:  # e == L is the suffix itself
                return self.ctx[e : e + k]
        return []

    def update(self, generated: list[int], accepted: int, drafted: int) -> None:
        pass


class PointerProposer:
    """PatchSprint alignment pointer into the source file.

    COPY:   draft source[ptr : ptr + k].
    EDIT:   after a rejection, the model is writing new code. Each step we try to
            RESYNC by matching the last n generated tokens (n from max_n down to
            min_n) against the source, preferring the closest match at or after
            ptr (edits usually move forward through the file). If nothing
            matches we draft nothing and fall back to plain decoding.
    """

    def __init__(self, source: list[int], min_n: int = 3, max_n: int = 8):
        self.source = source
        self.min_n, self.max_n = min_n, max_n
        self.ptr = 0
        self.mode = "COPY"
        # n-gram -> sorted list of end positions (exclusive) in source
        self.index: dict[int, dict[tuple[int, ...], list[int]]] = {}
        for n in range(min_n, max_n + 1):
            idx: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for e in range(n, len(source) + 1):
                idx[tuple(source[e - n : e])].append(e)
            self.index[n] = idx

    def _resync(self, generated: list[int]) -> int | None:
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(generated) < n:
                continue
            cands = self.index[n].get(tuple(generated[-n:]))
            if not cands:
                continue
            forward = [e for e in cands if e >= self.ptr]
            if forward:
                return min(forward)
            return max(cands)  # closest backward match (e.g. moved code)
        return None

    def propose(self, generated: list[int], k: int) -> list[int]:
        if self.mode == "EDIT":
            pos = self._resync(generated)
            if pos is None:
                return []
            self.ptr, self.mode = pos, "COPY"
        return self.source[self.ptr : self.ptr + k]

    def update(self, generated: list[int], accepted: int, drafted: int) -> None:
        if self.mode != "COPY":
            return
        self.ptr += accepted
        bonus = generated[-1]
        # The bonus token continues the copy only if it equals the next source token.
        if accepted == drafted and self.ptr < len(self.source) and self.source[self.ptr] == bonus:
            self.ptr += 1
        else:
            self.mode = "EDIT"


class HybridProposer(PointerProposer):
    """Pointer proposer that falls back to n-gram lookup over source + output.

    Inside newly written code the pointer has nothing to copy, but new code often
    reuses fragments from elsewhere in the file or from earlier generated lines.
    """

    def __init__(self, source: list[int], **kw):
        super().__init__(source, **kw)
        self.ngram = NgramProposer(source)

    def propose(self, generated: list[int], k: int) -> list[int]:
        draft = super().propose(generated, k)
        if self.mode == "EDIT":  # resync failed
            return self.ngram.propose(generated, k)
        return draft


# ---------------------------------------------------------------------------
# Per-problem analysis
# ---------------------------------------------------------------------------


def matching_blocks(a: list, b: list) -> list[tuple[int, int, int]]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(x.a, x.b, x.size) for x in sm.get_matching_blocks() if x.size > 0]


def analyze_problem(row, tok: Tokenizer, min_spans, ks) -> dict:
    before, after = row["before"], row["after"]
    enc_b = tok.encode(before, add_special_tokens=False)
    enc_a = tok.encode(after, add_special_tokens=False)
    src, out = enc_b.ids, enc_a.ids
    T = len(out)

    rec = {
        "id": int(row["id"]),
        "name": row["full_name"],
        "change_kind": row["taxonomy"]["change_kind"],
        "before_lines": before.count("\n") + 1,
        "after_lines": after.count("\n") + 1,
        "before_tokens": len(src),
        "after_tokens": T,
    }

    # Line-level edit ratio: fraction of output lines not copied from the input.
    a_lines, b_lines = before.splitlines(), after.splitlines()
    kept_lines = sum(s for _, _, s in matching_blocks(a_lines, b_lines))
    rec["edit_line_ratio"] = 1 - kept_lines / max(len(b_lines), 1)

    # Token-level copyable fraction for different minimum span lengths.
    blocks = matching_blocks(src, out)
    for m in min_spans:
        rec[f"copy_frac_m{m}"] = sum(s for _, _, s in blocks if s >= m) / max(T, 1)

    # Tokenization boundary loss: char-level copy coverage vs token-level coverage
    # (both measured in characters of the output).
    char_blocks = matching_blocks(list(before), list(after))
    char_cov = sum(s for _, _, s in char_blocks) / max(len(after), 1)
    offs = enc_a.offsets
    tok_cov_chars = sum(offs[b + s - 1][1] - offs[b][0] for _, b, s in blocks)
    rec["char_copy_frac"] = char_cov
    rec["tok_copy_frac_chars"] = tok_cov_chars / max(len(after), 1)
    rec["tokenization_loss"] = rec["char_copy_frac"] - rec["tok_copy_frac_chars"]

    # Replays.
    base = replay(out, NoSpecProposer(), k=1).steps
    assert base == T
    for k in ks:
        for name, prop in (
            ("ctx_oracle", ContextOracleProposer(src, out)),
            ("align_oracle", AlignOracleProposer(out, blocks)),
            ("ngram", NgramProposer(src)),
            ("pointer", PointerProposer(src)),
            ("hybrid", HybridProposer(src)),
        ):
            st = replay(out, prop, k)
            rec[f"{name}_k{k}_steps"] = st.steps
            rec[f"{name}_k{k}_speedup"] = T / st.steps
            rec[f"{name}_k{k}_accept_rate"] = st.acceptance_rate
            rec[f"{name}_k{k}_wasted"] = st.proposed - st.accepted
            rec[f"{name}_k{k}_proposed"] = st.proposed
    return rec


PROPOSERS = ("ctx_oracle", "align_oracle", "ngram", "pointer", "hybrid")


def summarize(df: pd.DataFrame, min_spans, ks) -> dict:
    s: dict = {"n_problems": len(df)}
    for col in ["before_lines", "after_tokens", "edit_line_ratio", "char_copy_frac",
                "tok_copy_frac_chars", "tokenization_loss"]:
        s[col] = {"median": float(df[col].median()), "mean": float(df[col].mean())}
    total_T = int(df["after_tokens"].sum())
    for m in min_spans:
        col = f"copy_frac_m{m}"
        weighted = float((df[col] * df["after_tokens"]).sum() / total_T)
        s[col] = {"median": float(df[col].median()), "token_weighted": weighted}
    for k in ks:
        for name in PROPOSERS:
            steps = int(df[f"{name}_k{k}_steps"].sum())
            s[f"{name}_k{k}"] = {
                "aggregate_step_speedup": total_T / steps,   # total tokens / total steps
                "median_step_speedup": float(df[f"{name}_k{k}_speedup"].median()),
                "mean_accept_rate": float(df[f"{name}_k{k}_accept_rate"].mean()),
                "wasted_draft_tokens": int(df[f"{name}_k{k}_wasted"].sum()),
                # tokens pushed through the target model per output token; 1.0 = plain decoding.
                # This is the relative compute cost once verification is compute-bound.
                "verified_tokens_per_output_token": (steps + int(df[f"{name}_k{k}_proposed"].sum())) / total_T,
            }
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="CanItEdit parquet file")
    ap.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json of the target model")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--min-spans", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    data = pd.read_parquet(args.data)
    records = [analyze_problem(r, tok, args.min_spans, args.ks) for _, r in data.iterrows()]
    df = pd.DataFrame(records).sort_values("id")

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_problem.csv", index=False)
    summary = summarize(df, args.min_spans, args.ks)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
