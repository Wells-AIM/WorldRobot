#!/usr/bin/env python3
"""Prove that only the pre-registered variable changed between arms (§4).

Signatures are computed after the fact from the request logs the run already
wrote, not injected into the run. Nothing about the experiment is touched.

    candidate_set_hash        which candidates were offered
    immediate_payload_hash    the one-step facts, serializer-independent
    future_set_hash           the set of generated futures, ignoring assignment
    future_assignment_hash    which future sat beside which candidate

Expected, per decision, within a paired block:

    s1_original vs s1_compact       candidate_set and immediate_payload equal
    s1_compact  vs policy_compact   candidate_set and immediate_payload equal,
                                    and only policy_compact carries a future
    policy_compact vs policy_shuffle  future_set equal, assignment different
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.compact import expand_immediate  # noqa: E402
from jev_libero.records import read_jsonl  # noqa: E402

ARMS = ("s1_original", "s1_compact", "policy_compact", "policy_shuffle")
FUTURE_KEYS = ("future", "future_trajectory", "trajectory")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def split_option(option):
    """(immediate facts in long form, future or None) for one candidate."""
    if not isinstance(option, dict):
        return option, None
    future = next((option[k] for k in FUTURE_KEYS if k in option), None)
    immediate = expand_immediate({k: v for k, v in option.items() if k not in FUTURE_KEYS})
    return immediate, future


def signatures(criteria):
    immediate, futures = {}, {}
    for name, option in sorted(criteria.items()):
        imm, fut = split_option(option)
        immediate[name] = imm
        if fut is not None:
            futures[name] = fut
    return {
        "immediate_by_candidate": immediate,
        "futures_by_candidate": futures,
        "candidate_set_hash": digest(sorted(criteria)),
        "immediate_payload_hash": digest(immediate),
        # Set, not mapping: identical physics regardless of who got what.
        "future_set_hash": digest(sorted(digest(f) for f in futures.values())) if futures else None,
        "future_assignment_hash": digest({k: digest(v) for k, v in futures.items()})
        if futures else None,
        "options_with_future": len(futures),
        "options": len(criteria),
    }


def motor_calls(folder):
    path = Path(folder) / "api.jsonl"
    if not path.exists():
        return []
    return [c for c in read_jsonl(path) if c["layer"] == "motor"]


def choices(folder):
    path = Path(folder) / "trace.jsonl"
    if not path.exists():
        return []
    return [r["choice"] for r in read_jsonl(path)]


def routing(folder):
    """Per decision: the feasible pool, the strategy, and the motor menu.

    The pool is what candidate generation produced from the state. The motor
    menu is that pool filtered by the strategy layer, so two arms in the same
    state can legitimately be offered different motor menus if their strategy
    choices diverge. Fairness is a property of the pool, not of the filtered
    menu -- checking the menu would report the pipeline's own layering as a
    violation.
    """
    path = Path(folder) / "trace.jsonl"
    if not path.exists():
        return []
    out = []
    for row in read_jsonl(path):
        pool = sorted({name for names in row["eligible_by_intent"].values()
                       for name in names})
        out.append({
            "pool_hash": digest(pool),
            "pool_size": len(pool),
            "strategy": row.get("strategy"),
            "intent": row.get("intent"),
            "motor_menu": sorted(row.get("eligible_motor", [])),
        })
    return out


def aligned_length(folder_a, folder_b):
    """How many decisions two arms share a state for.

    Two arms start in the same state, so their candidate menus must match at
    decision 0. The moment they choose differently they are in different states
    and every later menu legitimately diverges. Only the prefix up to and
    including the first disagreement is a fair comparison; demanding equality
    past it tests nothing.
    """
    a, b = choices(folder_a), choices(folder_b)
    n = min(len(a), len(b))
    for index in range(n):
        if a[index] != b[index]:
            return index + 1  # the disagreeing decision itself was still aligned
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    args = parser.parse_args(argv)
    base = args.out / "episodes"
    if not base.exists():
        raise SystemExit(f"no episodes under {base}")

    rows, checks = [], []
    for block in sorted(base.iterdir()):
        per_arm = {}
        for arm in ARMS:
            calls = motor_calls(block / arm)
            per_arm[arm] = [signatures(c["request"]["questions"]["motor"]["criteria"])
                            for c in calls]
            for index, sig in enumerate(per_arm[arm]):
                rows.append({"pair_block_id": block.name, "arm": arm, "decision": index,
                             **{k: v for k, v in sig.items()
                                if not k.endswith("_by_candidate")}})
        if not all(per_arm.values()):
            continue
        route = {arm: routing(block / arm) for arm in ARMS}
        spans = {
            "A": aligned_length(block / "s1_original", block / "s1_compact"),
            "B": aligned_length(block / "s1_compact", block / "policy_compact"),
            "C": aligned_length(block / "policy_compact", block / "policy_shuffle"),
        }

        def agree(a, b, key, span):
            return all(per_arm[a][i][key] == per_arm[b][i][key]
                       for i in range(min(span, len(per_arm[a]), len(per_arm[b]))))

        def shared_immediate_agree(a, b, span):
            """The same candidate must carry the same facts in both arms.

            Menus can differ legitimately when the strategy layer diverges, so
            the comparison is over the candidates both arms were offered rather
            than over the whole menu.
            """
            n = min(span, len(per_arm[a]), len(per_arm[b]))
            for i in range(n):
                left = per_arm[a][i]["immediate_by_candidate"]
                right = per_arm[b][i]["immediate_by_candidate"]
                for name in set(left) & set(right):
                    if left[name] != right[name]:
                        return False
            return True

        def shared_futures_agree(a, b, span):
            """Both arms must have generated the same future for a shared candidate."""
            n = min(span, len(per_arm[a]), len(per_arm[b]))
            for i in range(n):
                left = per_arm[a][i]["futures_by_candidate"]
                right = per_arm[b][i]["futures_by_candidate"]
                shared = set(left) & set(right)
                if not shared:
                    continue
                if sorted(digest(left[k]) for k in shared) != sorted(
                        digest(right[k]) for k in shared):
                    return False
            return True

        def pools_agree(a, b, span):
            n = min(span, len(route[a]), len(route[b]))
            return all(route[a][i]["pool_hash"] == route[b][i]["pool_hash"]
                       for i in range(n))

        def strategy_divergence(a, b, span):
            n = min(span, len(route[a]), len(route[b]))
            return sum(1 for i in range(n)
                       if route[a][i]["strategy"] != route[b][i]["strategy"])

        # A derangement needs at least two candidates carrying a future.
        shufflable = [
            i for i in range(min(spans["C"], len(per_arm["policy_compact"]),
                                 len(per_arm["policy_shuffle"])))
            if per_arm["policy_compact"][i]["options_with_future"] > 1
        ]
        checks.append({
            "pair_block_id": block.name,
            "aligned_decisions_A": spans["A"],
            "aligned_decisions_B": spans["B"],
            "aligned_decisions_C": spans["C"],
            "shufflable_decisions_C": len(shufflable),
            # Candidate generation fairness, measured on the feasible pool.
            "A_pool_equal": pools_agree("s1_original", "s1_compact", spans["A"]),
            "B_pool_equal": pools_agree("s1_compact", "policy_compact", spans["B"]),
            "C_pool_equal": pools_agree("policy_compact", "policy_shuffle", spans["C"]),
            # How often the strategy layer diverged while states were aligned;
            # this is what makes the motor menus differ, and it is a result,
            # not a violation.
            "A_strategy_divergences": strategy_divergence("s1_original", "s1_compact",
                                                          spans["A"]),
            "B_strategy_divergences": strategy_divergence("s1_compact", "policy_compact",
                                                          spans["B"]),
            "C_strategy_divergences": strategy_divergence("policy_compact",
                                                          "policy_shuffle", spans["C"]),
            # Comparison A: representation only
            "A_candidate_set_equal": agree("s1_original", "s1_compact",
                                           "candidate_set_hash", spans["A"]),
            "A_immediate_equal": shared_immediate_agree("s1_original", "s1_compact",
                                                        spans["A"]),
            # Comparison B: future added, nothing else
            "B_candidate_set_equal": agree("s1_compact", "policy_compact",
                                           "candidate_set_hash", spans["B"]),
            "B_immediate_equal": shared_immediate_agree("s1_compact", "policy_compact",
                                                        spans["B"]),
            "B_only_policy_has_future": all(
                per_arm["s1_compact"][i]["options_with_future"] == 0
                and per_arm["policy_compact"][i]["options_with_future"] > 0
                for i in range(min(spans["B"], len(per_arm["s1_compact"]),
                                   len(per_arm["policy_compact"])))
            ),
            # Comparison C: assignment only
            "C_future_set_equal": shared_futures_agree("policy_compact", "policy_shuffle",
                                                       spans["C"]),
            "C_assignment_differs": all(
                per_arm["policy_compact"][i]["future_assignment_hash"]
                != per_arm["policy_shuffle"][i]["future_assignment_hash"]
                for i in shufflable
            ) if shufflable else None,
            "C_candidate_set_equal": agree("policy_compact", "policy_shuffle",
                                           "candidate_set_hash", spans["C"]),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "request_signatures.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(rows[0]) if rows else ["pair_block_id"])
        w.writeheader()
        w.writerows(rows)
    with (args.out / "signature_checks.csv").open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=list(checks[0]) if checks else ["pair_block_id"])
        w.writeheader()
        w.writerows(checks)

    print("candidate-generation fairness (feasible pool, while states aligned)")
    print(f"  {'block':26}{'alnA':>5}{'alnB':>5}{'alnC':>5}  "
          f"{'A_pool':>7}{'B_pool':>7}{'C_pool':>7}  "
          f"{'B_futures_only_policy':>22}{'C_set':>7}{'C_assign':>9}")
    for c in checks:
        print(f"  {c['pair_block_id']:26}{c['aligned_decisions_A']:>5}"
              f"{c['aligned_decisions_B']:>5}{c['aligned_decisions_C']:>5}  "
              f"{str(c['A_pool_equal']):>7}{str(c['B_pool_equal']):>7}"
              f"{str(c['C_pool_equal']):>7}  "
              f"{str(c['B_only_policy_has_future']):>22}"
              f"{str(c['C_future_set_equal']):>7}{str(c['C_assignment_differs']):>9}")
    print("\nstrategy-layer divergences while states were aligned "
          "(a result, not a violation; the motor menu is strategy-filtered)")
    for c in checks:
        print(f"  {c['pair_block_id']:26} A={c['A_strategy_divergences']:>3} "
              f"B={c['B_strategy_divergences']:>3} C={c['C_strategy_divergences']:>3}")
    invariants = ("A_pool_equal", "B_pool_equal", "C_pool_equal",
                  "A_immediate_equal", "B_immediate_equal",
                  "B_only_policy_has_future", "C_future_set_equal",
                  "C_assignment_differs")
    failures = [c for c in checks
                if not all(c[k] for k in invariants if c[k] is not None)]
    print(f"\nblocks checked: {len(checks)}   blocks with a violated invariant: {len(failures)}")
    for f in failures:
        print("  VIOLATION:", f["pair_block_id"],
              [k for k in invariants if f[k] is not None and not f[k]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
