#!/usr/bin/env python3
"""Stage 1.5: six arms differing only in how the counterfactual future is built.

    S1                  strong one-step baseline (original_no_oracle)
    S1+Repeat           + trajectory, chunk = [a, a, a]
    S1+Hold             + trajectory, chunk = [a, hold, hold]
    S1+PolicyFull       + trajectory, chunk decided from each branch state
    S1+PolicyCompact    the same futures, reported as paths and transitions
    S1+PolicyShuffle    the same futures, deranged against their candidates

Every arm shares the task, seed, initial state, candidate generator, primitives,
hard feasibility rules, provider and decision cap. Defaults to the mock provider,
which is a pipeline check and NOT Jev.

    python tools/run_stage15.py --out runs/stage15 --provider mock
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.experiment import STAGE15_ARMS  # noqa: E402
from jev_libero.records import read_jsonl  # noqa: E402

COLUMNS = [
    "arm", "task", "seed", "init_state_id", "success", "decisions",
    "far", "mid", "near", "terminal",
    "tokens_per_decision_estimated", "future_tokens_per_decision_estimated",
    "reported_input_tokens_per_decision", "rollout_ms", "cost_usd",
    "environment_steps", "final_distance_mm", "target_contact_decisions",
    "action_40mm", "action_10mm", "action_3mm", "mean_progress_mm_per_decision",
    "termination_reason", "still_progressing_at_cap", "error",
]


def action_size(name):
    import re

    m = re.match(r"[xyz]([+-]\d+)mm", name)
    return abs(int(m.group(1))) if m else None


def summarise(folder, arm, task, seed, init_state):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    trace = read_jsonl(folder / "trace.jsonl") if (folder / "trace.jsonl").exists() else []
    preds = read_jsonl(folder / "predictions.jsonl") if (folder / "predictions.jsonl").exists() else []
    api = read_jsonl(folder / "api.jsonl") if (folder / "api.jsonl").exists() else []
    initial = json.loads((folder / "initial_state.json").read_text())
    final = json.loads((folder / "final_state.json").read_text()) if (folder / "final_state.json").exists() else {}
    decisions = max(summary.get("decisions", 0), 1)
    sizes = [action_size(r["choice"]) for r in trace]
    progress = [r["before_task_value"] - r["after_task_value"] for r in trace]
    reported = sum(
        c.get("response", {}).get("usage", {}).get("input_tokens", 0) for c in api
    )
    phases = summary.get("decisions_by_phase", {})
    return {
        "arm": arm,
        "task": task,
        "seed": seed,
        "init_state_id": init_state,
        "success": int(bool(summary.get("success"))),
        "decisions": summary.get("decisions"),
        "far": phases.get("far", 0),
        "mid": phases.get("mid", 0),
        "near": phases.get("near", 0),
        "terminal": phases.get("terminal", 0),
        "tokens_per_decision_estimated": summary.get("estimated_tokens_per_decision", 0),
        "future_tokens_per_decision_estimated": summary.get(
            "estimated_future_tokens_per_decision", 0
        ),
        "reported_input_tokens_per_decision": round(reported / decisions) if reported else 0,
        "rollout_ms": summary.get("rollout_latency_ms_total", 0.0),
        "cost_usd": summary.get("cost_usd", 0.0),
        "environment_steps": summary.get("sim_steps"),
        "final_distance_mm": round(final.get("remaining_open_mm", float("nan")), 3)
        if final else None,
        "target_contact_decisions": sum(1 for p in preds if p["before"]["moving_contact"]),
        "action_40mm": sum(1 for s in sizes if s == 40),
        "action_10mm": sum(1 for s in sizes if s == 10),
        "action_3mm": sum(1 for s in sizes if s == 3),
        "mean_progress_mm_per_decision": round(sum(progress) / decisions, 3),
        "termination_reason": summary.get("termination_reason", summary.get("termination")),
        "still_progressing_at_cap": int(bool(summary.get("still_progressing_at_cap"))),
        "error": summary.get("error"),
        "_initial": initial.get("remaining_open_mm"),
    }


def phase_breakdown(folder):
    """Decisions, progress rate and action mix per distance band."""
    trace = read_jsonl(Path(folder) / "trace.jsonl")
    bands = {}
    for row in trace:
        band = row.get("phase", "unknown")
        entry = bands.setdefault(band, {"n": 0, "progress": 0.0, 40: 0, 10: 0, 3: 0})
        entry["n"] += 1
        entry["progress"] += row["before_task_value"] - row["after_task_value"]
        size = action_size(row["choice"])
        if size in (40, 10, 3):
            entry[size] += 1
    for entry in bands.values():
        entry["mm_per_decision"] = round(entry["progress"] / max(entry["n"], 1), 2)
    return bands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="top_drawer")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(STAGE15_ARMS))
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--candidate-depth", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=27)
    # Stage-1 showed a cap taken from the fastest arm turns "slower" into "failed".
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--budget-usd", type=float, default=0.05)
    parser.add_argument("--provider", choices=("mock", "openrouter", "typesafe"), default="mock")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/stage15"))
    args = parser.parse_args(argv)

    if args.provider != "mock":
        print(f"WARNING: provider={args.provider} makes PAID calls for "
              f"{len(args.arms)} episodes at up to ${args.budget_usd} each.", file=sys.stderr)

    from jev_libero.runner import run

    args.out.mkdir(parents=True, exist_ok=True)
    rows, breakdowns = [], {}
    for arm in args.arms:
        spec = STAGE15_ARMS[arm]
        folder = args.out / arm.replace("+", "_")
        if not folder.exists():
            print(f"=== {arm}", flush=True)
            try:
                run(
                    task=args.task, out=folder, seed=args.seed, init_state=args.init_state,
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
                rows.append({**{c: None for c in COLUMNS}, "arm": arm, "task": args.task,
                             "error": traceback.format_exc(limit=1).strip()})
                continue
        else:
            print(f"skip existing {folder}", flush=True)
        row = summarise(folder, arm, args.task, args.seed, args.init_state)
        row.pop("_initial", None)
        rows.append(row)
        breakdowns[arm] = phase_breakdown(folder)

    csv_path = args.out / "stage15_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "stage15_phase_breakdown.json").write_text(
        json.dumps(breakdowns, indent=2) + "\n"
    )
    print(f"\nwrote {csv_path} ({len(rows)} arms)")
    print(f"wrote {args.out / 'stage15_phase_breakdown.json'}")
    if args.provider == "mock":
        print("\nMOCK PROVIDER: pipeline validation only. These are not Jev results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
