#!/usr/bin/env python3
"""Side-by-side of the three continuations at one state, across action scales.

Answers report section 5: does policy_consistent read the branch state, and how
does its future differ from repeat and hold for 40 / 10 / 3 mm inputs?

    python tools/continuation_diag.py --task top_drawer --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.continuation import make_continuation  # noqa: E402
from jev_libero.futures import CandidateSequence, chunk_actions, rollout  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="top_drawer")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prefix", nargs="*", default=[],
                        help="actions to advance the live state by first")
    parser.add_argument(
        "--inputs", nargs="+",
        default=["x+40mm", "x+10mm", "x+3mm", "x-40mm", "z-40mm", "y-40mm"],
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from jev_libero.world import Snapshot, World

    world = World(seed=args.seed, init_index=args.init_state, task_config=args.task)
    rows = []
    try:
        for action in args.prefix:
            world.execute(action, -1.0)
        root = Snapshot(world)
        policies = {name: make_continuation(name) for name in ("repeat", "hold",
                                                               "policy_consistent")}
        for first in args.inputs:
            entry = {"first_input": first}
            for name, policy in policies.items():
                root.restore()
                candidate = CandidateSequence(
                    candidate_id=first,
                    actions=chunk_actions(first, args.depth, "repeat"),
                    family="",
                    first_input=first,
                )
                result = rollout(world, candidate, -1.0, args.horizon, world.config,
                                 continuation=policy, depth=args.depth)
                gaps = [round(p["distance_to_moving_geometry_mm"], 2)
                        for p in result.checkpoints]
                joint = [round(p.get("remaining_open_mm", float("nan")), 2)
                         for p in result.checkpoints]
                entry[name] = {
                    "actions": result.actions,
                    "surface_gap_mm": gaps,
                    "articulation_mm": joint,
                    "ends_in_target_contact": bool(
                        result.checkpoints[-1]["moving_contact"]
                    ),
                    "reasons": [
                        {k: v for k, v in r.items() if k in ("step", "input", "rule", "effect")}
                        for r in result.continuation_reasons
                    ],
                }
            same = entry["policy_consistent"]["actions"] == entry["repeat"]["actions"]
            entry["policy_equals_repeat"] = same
            entry["endpoint_gap_difference_mm"] = round(
                entry["policy_consistent"]["surface_gap_mm"][-1]
                - entry["repeat"]["surface_gap_mm"][-1],
                2,
            )
            rows.append(entry)
        root.restore()
    finally:
        world.close()

    width = max(len(r["first_input"]) for r in rows) + 1
    print(f"{'input':{width}}{'repeat':28}{'hold':28}{'policy_consistent':28}{'Δend gap':>10}")
    for r in rows:
        print(
            f"{r['first_input']:{width}}"
            f"{','.join(a[:8] for a in r['repeat']['actions']):28}"
            f"{','.join(a[:8] for a in r['hold']['actions']):28}"
            f"{','.join(a[:8] for a in r['policy_consistent']['actions']):28}"
            f"{r['endpoint_gap_difference_mm']:>10}"
        )
    diverged = [r["first_input"] for r in rows if not r["policy_equals_repeat"]]
    print(f"\npolicy differs from repeat for {len(diverged)}/{len(rows)}: {diverged}")
    print("\nsurface gap paths (mm):")
    for r in rows:
        print(f"  {r['first_input']:8} repeat={r['repeat']['surface_gap_mm']}")
        print(f"  {'':8} policy={r['policy_consistent']['surface_gap_mm']}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
