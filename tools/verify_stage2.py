#!/usr/bin/env python3
"""Check the Stage 2 invariants on a completed run (§28)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.compact import expand_immediate  # noqa: E402
from jev_libero.futures import contains_oracle  # noqa: E402
from jev_libero.records import read_jsonl  # noqa: E402

ARMS = ("s1_original", "s1_compact", "policy_compact", "policy_shuffle")


def motor_calls(folder):
    path = folder / "api.jsonl"
    if not path.exists():
        return []
    return [c for c in read_jsonl(path) if c["layer"] == "motor"]


def main(base=Path("/tmp/s2mock2")):
    blocks = sorted((base / "episodes").iterdir())
    print(f"blocks: {[b.name for b in blocks]}\n")

    print("=== candidate fairness: same first-step menu across arms, per decision ===")
    all_same = True
    for block in blocks:
        menus = {}
        for arm in ARMS:
            calls = motor_calls(block / arm)
            menus[arm] = [sorted(c["request"]["questions"]["motor"]["criteria"]) for c in calls]
        n = min((len(v) for v in menus.values()), default=0)
        ok = all(menus[a][i] == menus["s1_original"][i] for a in ARMS for i in range(n))
        all_same &= ok
        print(f"  {block.name:26} decisions compared={n}  identical={ok}")
    print(f"  ALL BLOCKS IDENTICAL: {all_same}")

    print("\n=== information equivalence: s1_compact carries s1_original's facts ===")
    for block in blocks:
        a = motor_calls(block / "s1_original")
        b = motor_calls(block / "s1_compact")
        n = min(len(a), len(b))
        equal, checked = True, 0
        for i in range(n):
            ca = a[i]["request"]["questions"]["motor"]["criteria"]
            cb = b[i]["request"]["questions"]["motor"]["criteria"]
            if set(ca) != set(cb):
                equal = False
                break
            for name in ca:
                checked += 1
                if expand_immediate(cb[name]) != ca[name]:
                    equal = False
        print(f"  {block.name:26} options checked={checked:4} lossless={equal}")

    print("\n=== s1 arms carry no future ===")
    for block in blocks:
        for arm in ("s1_original", "s1_compact"):
            calls = motor_calls(block / arm)
            leaked = sum(
                1 for c in calls
                for o in c["request"]["questions"]["motor"]["criteria"].values()
                if isinstance(o, dict) and ({"future", "future_trajectory", "trajectory"} & set(o))
            )
            print(f"  {block.name:26} {arm:16} future fields = {leaked}")

    print("\n=== policy arms do carry a future ===")
    for block in blocks:
        for arm in ("policy_compact", "policy_shuffle"):
            calls = motor_calls(block / arm)
            carried = sum(
                1 for c in calls
                for o in c["request"]["questions"]["motor"]["criteria"].values()
                if isinstance(o, dict) and "future" in o
            )
            print(f"  {block.name:26} {arm:16} options with future = {carried}")

    print("\n=== oracle isolation (state + criteria only) ===")
    for arm in ARMS:
        hits = 0
        for block in blocks:
            for c in motor_calls(block / arm):
                hits += len(contains_oracle(c["request"]["questions"]["motor"]["criteria"]))
                hits += len(contains_oracle(c["request"]["state"]))
        print(f"  {arm:16} oracle hits = {hits}")

    print("\n=== shuffle: same physics, different assignment ===")
    for block in blocks:
        pc = block / "policy_compact" / "continuation.jsonl"
        ps = block / "policy_shuffle" / "continuation.jsonl"
        if not pc.exists() or not ps.exists():
            continue
        a, b = read_jsonl(pc), read_jsonl(ps)
        n = min(len(a), len(b))
        same_futures = all(
            {k: v["simulated_actions"] for k, v in a[i]["candidates"].items()}
            == {k: v["simulated_actions"] for k, v in b[i]["candidates"].items()}
            for i in range(n)
        )
        print(f"  {block.name:26} decisions={n} identical generated futures={same_futures}")

    print("\n=== arm order randomisation ===")
    rows = list(csv_rows(base / "episodes.csv"))
    orders = {}
    for r in rows:
        orders.setdefault(r["pair_block_id"], {})[int(r["arm_order_index"])] = r["arm"]
    for block, mapping in sorted(orders.items()):
        print(f"  {block:26} {[mapping[i] for i in sorted(mapping)]}")
    distinct = {tuple(m[i] for i in sorted(m)) for m in orders.values()}
    print(f"  distinct orders across blocks: {len(distinct)}")

    print("\n=== censoring fields present ===")
    keys = ("censored", "termination_reason", "still_progressing_at_cap",
            "last_5_decision_progress")
    missing = [k for k in keys if k not in rows[0]]
    print(f"  missing: {missing or 'none'}")
    print(f"  example: censored={rows[0]['censored']} "
          f"reason={rows[0]['termination_reason']} "
          f"progressing={rows[0]['still_progressing_at_cap']}")

    print("\n=== resume ===")
    meta = json.loads((base / "run_metadata.json").read_text())
    print(f"  mock_output={meta['mock_output']}  planned={meta['episodes_planned']}  "
          f"model_pinned={meta['model_version_pinned']}")
    return 0


def csv_rows(path):
    import csv

    with path.open() as stream:
        yield from csv.DictReader(stream)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/s2mock2")))
