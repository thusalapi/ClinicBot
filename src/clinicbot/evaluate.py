"""Measure the RAG system instead of eyeballing it.

Two levels:

1. Retrieval eval (fast, free, no LLM). For each question we know which
   section *should* be retrieved. Metrics:
     recall@k  fraction of questions whose relevant chunk is in the top k.
               If it's missing, the LLM never sees it, so no prompt can fix that.
     MRR       mean reciprocal rank: 1/rank of the first relevant chunk
               (rank 1 -> 1.0, rank 2 -> 0.5, missing -> 0). Rewards putting
               the right chunk *first*, not just somewhere in the list.

2. Answer eval (slow, runs the LLM). Checks that each answer contains a key
   fact (e.g. "500"). Crude, but cheap and deterministic. The next step up
   is "LLM-as-judge": asking a model to grade faithfulness and relevance.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

from clinicbot.config import PROJECT_ROOT
from clinicbot.retrieval import retrieve
from clinicbot.store import SearchResult

EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"


@dataclass
class Case:
    question: str
    relevant: list[str]  # "file.md::section substring"
    answer_must_include: list[str]
    type: str


def load_cases() -> list[Case]:
    with EVAL_FILE.open(encoding="utf-8") as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def is_relevant(result: SearchResult, labels: list[str]) -> bool:
    for label in labels:
        source, section = label.split("::", 1)
        if result.source == source and section in result.section:
            return True
    return False


def first_relevant_rank(results: list[SearchResult], labels: list[str]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if is_relevant(r, labels):
            return rank
    return None


def eval_retrieval(k: int, modes: list[str], show_misses: str | None) -> None:
    cases = [c for c in load_cases() if c.relevant]  # unanswerable ones have no target
    print(f"Retrieval eval: {len(cases)} questions, k={k}\n")

    types = sorted({c.type for c in cases})
    header = f"{'mode':<8} {'recall@k':>9} {'MRR':>6}   " + "  ".join(
        f"{t:>11}" for t in types
    )
    print(header)
    print("-" * len(header))

    for mode in modes:
        hits, rr = 0, 0.0
        by_type = defaultdict(lambda: [0, 0])  # type -> [hits, total]
        misses = []
        for case in cases:
            rank = first_relevant_rank(retrieve(case.question, k, mode), case.relevant)
            by_type[case.type][1] += 1
            if rank:
                hits += 1
                rr += 1 / rank
                by_type[case.type][0] += 1
            else:
                misses.append(case)
        per_type = "  ".join(
            f"{by_type[t][0]:>5}/{by_type[t][1]:<5}" for t in types
        )
        print(f"{mode:<8} {hits / len(cases):>9.0%} {rr / len(cases):>6.2f}   {per_type}")

        if show_misses == mode and misses:
            print(f"\n  Misses for {mode}:")
            for case in misses:
                got = retrieve(case.question, k, mode)
                print(f"   - {case.question}")
                print(f"       wanted {case.relevant[0]}")
                print(f"       got    {[f'{r.source}::{r.section.split(' > ')[-1]}' for r in got]}")
            print()
    print("\n(per-type columns are recall@k hits/total)")


def eval_answers(k: int, mode: str) -> None:
    from clinicbot.generate import answer

    cases = load_cases()
    print(f"Answer eval: {len(cases)} questions, mode={mode}, k={k}\n")
    passed = 0
    for i, case in enumerate(cases, start=1):
        reply = answer(case.question, retrieve(case.question, k, mode))
        # Each fact may list alternatives separated by "|", e.g. "not|don't".
        ok = all(
            any(alt.lower() in reply.lower() for alt in fact.split("|"))
            for fact in case.answer_must_include
        )
        passed += ok
        print(f"[{i:>2}/{len(cases)}] {'PASS' if ok else 'FAIL'}  {case.question}")
        if not ok:
            print(f"         expected to contain {case.answer_must_include}")
            print("         got: " + reply.replace("\n", " ")[:300])
    print(f"\nAnswer accuracy: {passed}/{len(cases)} = {passed / len(cases):.0%}")
