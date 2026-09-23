#!/usr/bin/env python3
"""Stage 2 pilot: separate representation, future information and correspondence.

Four arms, paired on (task, initial state). Each pair block runs all four, so a
difference is read within a block rather than across tasks.

    s1_original      accurate one-step consequence, published formatting
    s1_compact       the same facts, compact formatting, no future
    policy_compact   the same compact formatting, plus a policy-consistent future
    policy_shuffle   the same futures, deranged against their candidates

S1Original → S1Compact is the representation effect. S1Compact → PolicyCompact is
the marginal value of a correct multi-step future. PolicyCompact → PolicyShuffle
is the value of the action-future correspondence.

Arm order is permuted per block so provider drift cannot align with a method.
Every episode is checkpointed; --resume skips completed work.

    python tools/run_stage2_pilot.py --provider mock --init-states 0 1 --dry-run
    python tools/run_stage2_pilot.py --provider typesafe --api-key-file KEY
"""

import argparse
import csv
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.experiment import STAGE2_ARMS, STAGE2_MAIN_ARMS  # noqa: E402
from jev_libero.records import read_jsonl  # noqa: E402

TASKS = ("top_drawer", "microwave", "alphabet_soup")
BUDGET_MILESTONES = (10, 20, 30, 40, 50, 60)


def block_id(task, init_state):
    return f"{task}__init_{init_state:03d}"


def arm_order(task, init_state, seed):
    """Deterministic per-block permutation, so a method never owns a time slot."""
    order = list(STAGE2_MAIN_ARMS)
    random.Random(f"{seed}:{task}:{init_state}").shuffle(order)
    return order


def episode_dir(out, task, init_state, arm):
    return out / "episodes" / block_id(task, init_state) / arm


def is_complete(folder):
    """A finished episode has a summary and was not interrupted mid-write."""
    summary = folder / "summary.json"
    if not summary.exists():
        return False
    try:
        json.loads(summary.read_text())
        return True
    except (ValueError, OSError):
        return False


def decisions_to_success(trace):
    """First decision index at which the task became successful, else None."""
    for row in trace:
        if row.get("success"):
            return row["step"] + 1
    return None


def collect(folder, task, init_state, arm, provider, order_index, started, elapsed):
    summary = json.loads((folder / "summary.json").read_text())
    trace = read_jsonl(folder / "trace.jsonl") if (folder / "trace.jsonl").exists() else []
    api = read_jsonl(folder / "api.jsonl") if (folder / "api.jsonl").exists() else []
    final = {}
    if (folder / "final_state.json").exists():
        final = json.loads((folder / "final_state.json").read_text())
    config = json.loads((folder / "config.json").read_text())
    reached = decisions_to_success(trace)
    censored = not summary.get("success")
    tokens = sum(c.get("response", {}).get("usage", {}).get("input_tokens", 0) for c in api)
    decisions = max(summary.get("decisions", 0), 1)
    budget = {"base": 0, "immediate": 0, "future": 0}
    for row in trace:
        b = row.get("token_budget_estimated", {})
        budget["base"] += b.get("base_tokens", 0)
        budget["immediate"] += b.get("immediate_tokens", 0)
        budget["future"] += b.get("future_tokens", 0)
    phases = summary.get("decisions_by_phase", {})
    return {
        "pair_block_id": block_id(task, init_state),
        "task": task,
        "init_state_id": init_state,
        "arm": arm,
        "arm_order_index": order_index,
        "provider": provider,
        "requested_model": config.get("provider"),
        "run_started_utc": started,
        "runtime_s": round(elapsed, 1),
        "success": int(bool(summary.get("success"))),
        "decisions_to_success": reached,
        "censored": int(censored),
        "decisions": summary.get("decisions"),
        "termination_reason": summary.get("termination_reason", summary.get("termination")),
        "still_progressing_at_cap": int(bool(summary.get("still_progressing_at_cap"))),
        "last_5_decision_progress": json.dumps(summary.get("last_n_decision_progress", [])),
        "environment_steps": summary.get("sim_steps"),
        "jev_calls": summary.get("api_calls"),
        "input_tokens_total": tokens,
        "input_tokens_per_decision": round(tokens / decisions) if tokens else 0,
        "est_base_tokens_per_decision": round(budget["base"] / decisions),
        "est_immediate_tokens_per_decision": round(budget["immediate"] / decisions),
        "est_future_tokens_per_decision": round(budget["future"] / decisions),
        "cost_usd": summary.get("cost_usd", 0.0),
        "jev_latency_s": round(sum(r.get("decision_seconds", 0.0) for r in trace), 2),
        "rollout_latency_ms": summary.get("rollout_latency_ms_total", 0.0),
        "rollout_steps": summary.get("rollout_steps_total", 0),
        "far": phases.get("far", 0),
        "mid": phases.get("mid", 0),
        "near": phases.get("near", 0),
        "terminal": phases.get("terminal", 0),
        "final_task_value": summary.get("final_task_value"),
        "final_distance_mm": final.get("remaining_open_mm"),
        "error": summary.get("error"),
    }


def completion_curve(rows):
    """Success@N for the shared milestones, per arm and per task."""
    out = []
    arms = sorted({r["arm"] for r in rows})
    for scope, key in (("overall", None), ("task", "task")):
        groups = {None: rows} if key is None else {
            t: [r for r in rows if r["task"] == t] for t in sorted({r["task"] for r in rows})
        }
        for group, subset in groups.items():
            for arm in arms:
                mine = [r for r in subset if r["arm"] == arm]
                if not mine:
                    continue
                for n in BUDGET_MILESTONES:
                    hit = sum(
                        1 for r in mine
                        if r["decisions_to_success"] and r["decisions_to_success"] <= n
                    )
                    out.append({
                        "scope": scope, "group": group or "all", "arm": arm,
                        "budget": n, "episodes": len(mine), "successes": hit,
                        "rate": round(hit / len(mine), 4),
                    })
    return out


def write_csv(path, rows, columns=None):
    if not rows:
        path.write_text("")
        return
    columns = columns or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--init-states", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=list(STAGE2_MAIN_ARMS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--budget-usd", type=float, default=0.05)
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--candidate-depth", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=27)
    parser.add_argument("--arm-order-seed", type=int, default=20260923)
    parser.add_argument("--randomize-arm-order", action="store_true", default=True)
    parser.add_argument("--no-randomize-arm-order", dest="randomize_arm_order",
                        action="store_false")
    parser.add_argument("--provider", choices=("mock", "openrouter", "typesafe"),
                        default="mock")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and write no episodes")
    parser.add_argument("--resume", action="store_true",
                        help="skip episodes that already completed")
    parser.add_argument("--only-task")
    parser.add_argument("--only-init", type=int)
    parser.add_argument("--only-arm")
    parser.add_argument("--include-full-diagnostic", action="store_true",
                        help="also run policy_full, for the representation-dilution check")
    args = parser.parse_args(argv)

    arms = list(args.arms)
    if args.include_full_diagnostic and "policy_full" not in arms:
        arms.append("policy_full")
    if args.only_arm:
        arms = [args.only_arm]
    tasks = [args.only_task] if args.only_task else args.tasks
    inits = [args.only_init] if args.only_init is not None else args.init_states

    plan = []
    for task in tasks:
        for init_state in inits:
            order = (arm_order(task, init_state, args.arm_order_seed)
                     if args.randomize_arm_order else list(STAGE2_MAIN_ARMS))
            order = [a for a in order if a in arms] + [a for a in arms if a not in order]
            for index, arm in enumerate(order):
                plan.append((task, init_state, arm, index))

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"planned episodes: {len(plan)}  "
          f"({len(tasks)} tasks x {len(inits)} init states x {len(arms)} arms)")
    if args.dry_run:
        for task, init_state, arm, index in plan:
            folder = episode_dir(args.out, task, init_state, arm)
            state = "done" if is_complete(folder) else "todo"
            print(f"  {block_id(task, init_state):28} #{index} {arm:16} {state}")
        return 0

    if args.provider != "mock":
        print(f"WARNING: provider={args.provider} makes PAID calls for up to "
              f"{len(plan)} episodes at ${args.budget_usd} each.", file=sys.stderr)

    from jev_libero.runner import run

    rows, skipped = [], 0
    for task, init_state, arm, index in plan:
        folder = episode_dir(args.out, task, init_state, arm)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if is_complete(folder):
            if args.resume:
                skipped += 1
                rows.append(collect(folder, task, init_state, arm, args.provider,
                                    index, started, 0.0))
                continue
            print(f"exists, not resuming: {folder}", file=sys.stderr)
            rows.append(collect(folder, task, init_state, arm, args.provider,
                                index, started, 0.0))
            continue
        spec = STAGE2_ARMS[arm]
        print(f"=== {block_id(task, init_state)} #{index} {arm}", flush=True)
        began = time.perf_counter()
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
            rows.append(collect(folder, task, init_state, arm, args.provider, index,
                                started, time.perf_counter() - began))
        except Exception:
            traceback.print_exc()
            rows.append({
                "pair_block_id": block_id(task, init_state), "task": task,
                "init_state_id": init_state, "arm": arm, "arm_order_index": index,
                "provider": args.provider, "run_started_utc": started,
                "success": 0, "censored": 1,
                "termination_reason": "runner_exception",
                "error": traceback.format_exc(limit=1).strip(),
            })
        # Checkpoint after every episode so a crash never costs completed work.
        write_csv(args.out / "episodes.csv", rows)

    write_csv(args.out / "episodes.csv", rows)
    write_csv(args.out / "completion_curve.csv", completion_curve(
        [r for r in rows if r.get("decisions")]))
    (args.out / "run_metadata.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": tasks, "init_states": inits, "arms": arms,
        "seed": args.seed, "arm_order_seed": args.arm_order_seed,
        "randomize_arm_order": args.randomize_arm_order,
        "max_decisions": args.max_decisions, "horizon_s": args.horizon,
        "candidate_depth": args.candidate_depth, "candidate_count": args.candidate_count,
        "provider": args.provider,
        "requested_model": "jev-latest" if args.provider == "typesafe" else args.provider,
        "model_version_pinned": False,
        "reproducibility_note": (
            "The provider exposes only a rolling 'latest' alias; the resolved model "
            "version is not returned in the response, so runs are reproducible in "
            "configuration but not in model version."
        ),
        "episodes_planned": len(plan), "episodes_skipped_resume": skipped,
        "mock_output": args.provider == "mock",
    }, indent=2) + "\n")
    print(f"\nwrote {args.out / 'episodes.csv'} ({len(rows)} rows, {skipped} resumed)")
    if args.provider == "mock":
        print("MOCK PROVIDER: pipeline validation only. NOT REAL JEV RESULTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
