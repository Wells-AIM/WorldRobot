#!/usr/bin/env python3
"""Stage-1 controlled pilot: four future-information arms on the bundled tasks.

Every arm shares the task, the initial state, the seed, the candidate generator,
the candidate count, the action primitives, the hard feasibility rules, and the
decision budget. Only the future information given to the critic differs.

Defaults to the deterministic mock provider so a sweep costs nothing. Runs made
with --provider mock are pipeline checks, NOT Jev results.

    python tools/run_cfjev_pilot.py \
        --tasks microwave top_drawer alphabet_soup \
        --modes reactive short_preview counterfactual_future shuffle_future \
        --init-state-ids 0 1 2 --horizon 1.2 --out runs/pilot
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.experiment import CONTROLLED_MODES  # noqa: E402
from jev_libero.records import read_jsonl  # noqa: E402

COLUMNS = [
    "task",
    "seed",
    "init_state_id",
    "mode",
    "future_horizon_requested_s",
    "future_horizon_actual_s",
    "candidate_count",
    "candidate_depth",
    "success",
    "termination",
    "decision_count",
    "environment_steps",
    "collision_count",
    "forbidden_contact_count",
    "grasp_loss_count",
    "rollout_steps",
    "rollout_latency_ms",
    "jev_latency_ms",
    "total_latency_ms",
    "api_calls",
    "api_cost_usd",
    "provider",
    "error",
]


def episode_row(folder, task, seed, init_state, mode, provider):
    summary = json.loads((folder / "summary.json").read_text())
    trace = read_jsonl(folder / "trace.jsonl") if (folder / "trace.jsonl").exists() else []
    counterfactual = (
        read_jsonl(folder / "counterfactual.jsonl")
        if (folder / "counterfactual.jsonl").exists()
        else []
    )
    events = [event for row in counterfactual for c in row["candidates"]
              if c["counterfactual"] for event in c["counterfactual"]["events"]]
    return {
        "task": task,
        "seed": seed,
        "init_state_id": init_state,
        "mode": mode,
        "future_horizon_requested_s": summary.get("future_horizon_requested_s"),
        "future_horizon_actual_s": max(
            (row.get("future_horizon_actual_s") or 0.0 for row in trace), default=0.0
        ),
        "candidate_count": summary.get("candidate_count"),
        "candidate_depth": summary.get("candidate_depth"),
        "success": int(bool(summary["success"])),
        "termination": summary["termination"],
        "decision_count": summary["decisions"],
        "environment_steps": summary["sim_steps"],
        # Counted in the real executed episode, not inside counterfactual branches.
        "collision_count": sum(
            1 for row in trace if row.get("obstacle_contact_after")
        ),
        "forbidden_contact_count": sum(
            1 for event in events if event["event"] == "non_target_contact"
        ),
        "grasp_loss_count": sum(
            1 for event in events if event["event"] == "target_contact_lost"
        ),
        "rollout_steps": summary.get("rollout_steps_total", 0),
        "rollout_latency_ms": summary.get("rollout_latency_ms_total", 0.0),
        "jev_latency_ms": round(
            sum(row.get("decision_seconds", 0.0) for row in trace) * 1000, 3
        ),
        "total_latency_ms": round(
            sum(
                row.get("decision_seconds", 0.0) + row.get("prediction_seconds", 0.0)
                for row in trace
            )
            * 1000,
            3,
        ),
        "api_calls": summary["api_calls"],
        "api_cost_usd": summary["cost_usd"],
        "provider": provider,
        "error": summary.get("error"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["microwave", "top_drawer", "alphabet_soup"])
    parser.add_argument("--modes", nargs="+", default=list(CONTROLLED_MODES))
    parser.add_argument("--init-state-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--candidate-depth", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--budget-usd", type=float, default=0.02)
    parser.add_argument("--provider", choices=("mock", "openrouter", "typesafe"), default="mock")
    parser.add_argument("--out", type=Path, default=Path("runs/pilot"))
    args = parser.parse_args(argv)

    if args.provider != "mock":
        print(
            f"WARNING: provider={args.provider} makes PAID API calls for "
            f"{len(args.tasks) * len(args.modes) * len(args.init_state_ids) * len(args.seeds)} "
            "episodes.",
            file=sys.stderr,
        )

    from jev_libero.runner import run

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        for seed in args.seeds:
            for init_state in args.init_state_ids:
                for mode in args.modes:
                    folder = args.out / f"{task}_s{seed}_i{init_state}_{mode}"
                    if folder.exists():
                        print(f"skip existing {folder}", flush=True)
                        rows.append(
                            episode_row(folder, task, seed, init_state, mode, args.provider)
                        )
                        continue
                    print(f"=== {task} seed={seed} init={init_state} mode={mode}", flush=True)
                    try:
                        run(
                            task=task,
                            out=folder,
                            seed=seed,
                            init_state=init_state,
                            max_decisions=args.max_decisions,
                            budget_usd=args.budget_usd,
                            render=False,
                            provider=args.provider,
                            mode=mode,
                            candidate_count=args.candidate_count,
                            candidate_depth=args.candidate_depth,
                            future_horizon_s=args.horizon,
                        )
                        rows.append(
                            episode_row(folder, task, seed, init_state, mode, args.provider)
                        )
                    except Exception:
                        traceback.print_exc()
                        rows.append(
                            {
                                **{column: None for column in COLUMNS},
                                "task": task,
                                "seed": seed,
                                "init_state_id": init_state,
                                "mode": mode,
                                "provider": args.provider,
                                "error": traceback.format_exc(limit=1).strip(),
                            }
                        )

    csv_path = args.out / "pilot_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "pilot_summary.json").write_text(
        json.dumps(
            {"provider": args.provider, "mock_output": args.provider == "mock", "episodes": rows},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {csv_path} ({len(rows)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
