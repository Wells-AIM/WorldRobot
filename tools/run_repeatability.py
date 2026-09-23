#!/usr/bin/env python3
"""Repeatability probe: how much does Jev vary when nothing else does?

Runs `policy_compact` three times per task on a predeclared initial state, with
identical task, seed, initial state and prompt construction. Only the provider's
own stochasticity can differ.

Reports only. It must not change the main design — whether the main study needs
repeated inference is a decision to take after reading this, not automatically.

    python tools/run_repeatability.py --provider typesafe --api-key-file KEY
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.experiment import STAGE2_ARMS  # noqa: E402

# Declared before the run, not chosen from results.
PROBE_STATES = {"top_drawer": 0, "microwave": 0, "alphabet_soup": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(PROBE_STATES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arm", default="policy_compact")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--budget-usd", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--candidate-depth", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=27)
    parser.add_argument("--provider", choices=("mock", "openrouter", "typesafe"),
                        default="mock")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    from jev_libero.runner import run

    spec = STAGE2_ARMS[args.arm]
    base = args.out / "repeatability"
    base.mkdir(parents=True, exist_ok=True)
    (base / "probe_states.json").write_text(json.dumps(
        {"declared_before_run": True, "states": PROBE_STATES, "arm": args.arm,
         "repeats": args.repeats}, indent=2) + "\n")

    for task in args.tasks:
        init_state = PROBE_STATES[task]
        for index in range(args.repeats):
            folder = base / task / f"repeat_{index}"
            if (folder / "summary.json").exists():
                if args.resume:
                    print(f"resume: skip {folder}", flush=True)
                    continue
                print(f"exists: {folder}", file=sys.stderr)
                continue
            print(f"=== {task} init={init_state} repeat {index}", flush=True)
            try:
                folder.parent.mkdir(parents=True, exist_ok=True)
                run(
                    task=task, out=folder, seed=args.seed, init_state=init_state,
                    max_decisions=args.max_decisions, budget_usd=args.budget_usd,
                    render=False, provider=args.provider, key_file=args.api_key_file,
                    mode=spec["mode"],
                    candidate_count=args.candidate_count,
                    candidate_depth=args.candidate_depth,
                    future_horizon_s=args.horizon,
                    chunk_continuation=spec.get("continuation", "repeat"),
                    future_representation=spec.get("representation", "full"),
                    shuffle_future_arm=bool(spec.get("shuffle")),
                )
            except Exception:
                traceback.print_exc()
    print("\nrepeatability episodes written under", base)
    if args.provider == "mock":
        print("MOCK PROVIDER: NOT REAL JEV RESULTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
