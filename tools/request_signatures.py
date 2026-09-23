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
                rows.append({"pair_block_id": block.name, "arm": arm,
                             "decision": index, **sig})
        n = min((len(v) for v in per_arm.values() if v), default=0)
        if not n:
            continue

        def agree(a, b, key):
            return all(per_arm[a][i][key] == per_arm[b][i][key] for i in range(n))

        checks.append({
            "pair_block_id": block.name,
            "decisions_compared": n,
            # Comparison A: representation only
            "A_candidate_set_equal": agree("s1_original", "s1_compact", "candidate_set_hash"),
            "A_immediate_equal": agree("s1_original", "s1_compact", "immediate_payload_hash"),
            # Comparison B: future added, nothing else
            "B_candidate_set_equal": agree("s1_compact", "policy_compact", "candidate_set_hash"),
            "B_immediate_equal": agree("s1_compact", "policy_compact", "immediate_payload_hash"),
            "B_only_policy_has_future": all(
                per_arm["s1_compact"][i]["options_with_future"] == 0
                and per_arm["policy_compact"][i]["options_with_future"] > 0
                for i in range(n)
            ),
            # Comparison C: assignment only
            "C_future_set_equal": agree("policy_compact", "policy_shuffle", "future_set_hash"),
            "C_assignment_differs": all(
                per_arm["policy_compact"][i]["future_assignment_hash"]
                != per_arm["policy_shuffle"][i]["future_assignment_hash"]
                for i in range(n)
            ),
            "C_candidate_set_equal": agree("policy_compact", "policy_shuffle",
                                           "candidate_set_hash"),
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

    print(f"{'block':28}{'dec':>4}  A_cand A_imm  B_cand B_imm B_fut  C_set C_assign C_cand")
    for c in checks:
        print(f"{c['pair_block_id']:28}{c['decisions_compared']:>4}  "
              f"{str(c['A_candidate_set_equal']):>6} {str(c['A_immediate_equal']):>5}  "
              f"{str(c['B_candidate_set_equal']):>6} {str(c['B_immediate_equal']):>5} "
              f"{str(c['B_only_policy_has_future']):>5}  "
              f"{str(c['C_future_set_equal']):>5} {str(c['C_assignment_differs']):>8} "
              f"{str(c['C_candidate_set_equal']):>6}")
    failures = [c for c in checks if not all(v for k, v in c.items()
                                             if k.startswith(("A_", "B_", "C_")))]
    print(f"\nblocks checked: {len(checks)}   blocks with a violated invariant: {len(failures)}")
    for f in failures:
        print("  VIOLATION:", f["pair_block_id"],
              [k for k, v in f.items() if k.startswith(("A_", "B_", "C_")) and not v])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
