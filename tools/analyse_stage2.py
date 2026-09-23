#!/usr/bin/env python3
"""Stage 2 analysis: the three paired comparisons, plus mechanism diagnostics.

Comparison 1  s1_original  -> s1_compact      representation effect
Comparison 2  s1_compact   -> policy_compact  marginal value of a correct future
Comparison 3  policy_compact -> policy_shuffle  value of the correspondence

Everything here is paired within a (task, initial state) block and reported as
descriptive pilot output. Bootstrap intervals are given for the paired
differences; they are not significance tests.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.records import read_jsonl  # noqa: E402

COMPARISONS = [
    ("representation", "s1_original", "s1_compact"),
    ("future_information", "s1_compact", "policy_compact"),
    ("correspondence", "policy_compact", "policy_shuffle"),
]
CAP_SENTINEL = None  # censored episodes never become "cap decisions to success"


def load_episodes(out):
    path = out / "episodes.csv"
    if not path.exists():
        raise SystemExit(f"no episodes.csv under {out}; run the pilot first")
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["success"] = int(r["success"] or 0)
        r["censored"] = int(r["censored"] or 0)
        r["decisions"] = int(r["decisions"]) if r.get("decisions") else None
        d = r.get("decisions_to_success")
        r["decisions_to_success"] = int(d) if d not in (None, "", "None") else CAP_SENTINEL
        for key in ("input_tokens_per_decision", "est_future_tokens_per_decision",
                    "est_immediate_tokens_per_decision"):
            r[key] = int(r[key]) if r.get(key) else 0
        r["cost_usd"] = float(r["cost_usd"]) if r.get("cost_usd") else 0.0
    return rows


def bootstrap_ci(values, rounds=2000, seed=7):
    """Percentile interval on the mean. Descriptive, not a test."""
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (round(means[int(0.025 * rounds)], 3), round(means[int(0.975 * rounds)], 3))


def paired_rows(rows, arm_a, arm_b):
    """Blocks where both arms produced an episode."""
    by_block = {}
    for r in rows:
        by_block.setdefault(r["pair_block_id"], {})[r["arm"]] = r
    pairs = []
    for block, arms in sorted(by_block.items()):
        if arm_a in arms and arm_b in arms:
            pairs.append((block, arms[arm_a], arms[arm_b]))
    return pairs


def compare(rows, label, arm_a, arm_b):
    pairs = paired_rows(rows, arm_a, arm_b)
    both_solved = [(b, a, c) for b, a, c in pairs if a["success"] and c["success"]]
    deltas = [c["decisions_to_success"] - a["decisions_to_success"] for _, a, c in both_solved]
    return {
        "comparison": label,
        "baseline": arm_a,
        "variant": arm_b,
        "paired_blocks": len(pairs),
        "baseline_successes": sum(a["success"] for _, a, _ in pairs),
        "variant_successes": sum(c["success"] for _, _, c in pairs),
        "blocks_both_solved": len(both_solved),
        "variant_faster": sum(1 for d in deltas if d < 0),
        "variant_slower": sum(1 for d in deltas if d > 0),
        "tied": sum(1 for d in deltas if d == 0),
        "mean_decision_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "delta_ci_low": bootstrap_ci(deltas)[0],
        "delta_ci_high": bootstrap_ci(deltas)[1],
        "note": "descriptive pilot output; not a significance test",
    }


def action_disagreement(out, rows):
    """Where s1_compact and policy_compact chose differently, and what followed.

    Only the first divergence per block is comparable: after it the two arms are
    in different states, so later decisions are not aligned.
    """
    records = []
    for block, a, b in paired_rows(rows, "s1_compact", "policy_compact"):
        folder_a = out / "episodes" / block / "s1_compact"
        folder_b = out / "episodes" / block / "policy_compact"
        if not (folder_a / "trace.jsonl").exists() or not (folder_b / "trace.jsonl").exists():
            continue
        trace_a = read_jsonl(folder_a / "trace.jsonl")
        trace_b = read_jsonl(folder_b / "trace.jsonl")
        cont = {}
        if (folder_b / "continuation.jsonl").exists():
            for row in read_jsonl(folder_b / "continuation.jsonl"):
                cont[row["step"]] = row
        aligned = True
        for step, (ra, rb) in enumerate(zip(trace_a, trace_b)):
            same = ra["choice"] == rb["choice"]
            novelty = None
            if step in cont:
                chosen = cont[step]["candidates"].get(rb["choice"], {})
                novelty = chosen.get("future_novelty_count")
            records.append({
                "pair_block_id": block,
                "step": step,
                "states_aligned": int(aligned),
                "s1_compact_choice": ra["choice"],
                "policy_compact_choice": rb["choice"],
                "agreement": int(same),
                "s1_before_task_value": round(ra["before_task_value"], 3),
                "policy_before_task_value": round(rb["before_task_value"], 3),
                "s1_progress_mm": round(ra["before_task_value"] - ra["after_task_value"], 3),
                "policy_progress_mm": round(rb["before_task_value"] - rb["after_task_value"], 3),
                "policy_future_novelty_count": novelty,
                "note": "no good/bad label is assigned; mechanism analysis only",
            })
            if not same:
                aligned = False  # states diverge from here on
        # keep going past divergence but flagged, so the table shows both regimes
    return records


def repeatability(out):
    """Variation across identical repeats of policy_compact."""
    base = out / "repeatability"
    if not base.exists():
        return []
    rows = []
    groups = {}
    for folder in sorted(base.glob("*/*")):
        if not (folder / "summary.json").exists():
            continue
        task = folder.parent.name
        summary = json.loads((folder / "summary.json").read_text())
        trace = read_jsonl(folder / "trace.jsonl") if (folder / "trace.jsonl").exists() else []
        api = read_jsonl(folder / "api.jsonl") if (folder / "api.jsonl").exists() else []
        tokens = sum(c.get("response", {}).get("usage", {}).get("input_tokens", 0) for c in api)
        entry = {
            "task": task,
            "repeat": folder.name,
            "success": int(bool(summary.get("success"))),
            "decisions": summary.get("decisions"),
            "first_action": trace[0]["choice"] if trace else None,
            "action_sequence": [r["choice"] for r in trace],
            "input_tokens": tokens,
            "cost_usd": summary.get("cost_usd", 0.0),
        }
        groups.setdefault(task, []).append(entry)
    for task, entries in sorted(groups.items()):
        firsts = {e["first_action"] for e in entries}
        # Sequences only stay comparable while the states agree.
        prefix = 0
        if len(entries) > 1:
            sequences = [e["action_sequence"] for e in entries]
            for index in range(min(len(s) for s in sequences)):
                if len({s[index] for s in sequences}) == 1:
                    prefix += 1
                else:
                    break
        decisions = [e["decisions"] for e in entries if e["decisions"]]
        tokens = [e["input_tokens"] for e in entries]
        costs = [e["cost_usd"] for e in entries]
        rows.append({
            "task": task,
            "repeats": len(entries),
            "successes": sum(e["success"] for e in entries),
            "first_action_agreement": int(len(firsts) == 1),
            "identical_action_prefix": prefix,
            "decisions_min": min(decisions) if decisions else None,
            "decisions_max": max(decisions) if decisions else None,
            "decisions_spread": (max(decisions) - min(decisions)) if decisions else None,
            "tokens_min": min(tokens), "tokens_max": max(tokens),
            "tokens_spread_pct": round(
                100 * (max(tokens) - min(tokens)) / max(max(tokens), 1), 2),
            "cost_min": round(min(costs), 6), "cost_max": round(max(costs), 6),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                             for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/stage2"))
    args = parser.parse_args(argv)
    rows = load_episodes(args.out)

    pairs = [compare(rows, *c) for c in COMPARISONS]
    write_csv(args.out / "pairs.csv", pairs)

    tasks = {}
    for r in rows:
        key = (r["task"], r["arm"])
        entry = tasks.setdefault(key, {"task": r["task"], "arm": r["arm"], "episodes": 0,
                                       "successes": 0, "decisions": [], "tokens": [],
                                       "cost": 0.0, "censored": 0})
        entry["episodes"] += 1
        entry["successes"] += r["success"]
        entry["censored"] += r["censored"]
        if r["decisions_to_success"]:
            entry["decisions"].append(r["decisions_to_success"])
        entry["tokens"].append(r["input_tokens_per_decision"])
        entry["cost"] += r["cost_usd"]
    task_rows = []
    for entry in tasks.values():
        d = entry.pop("decisions")
        t = entry.pop("tokens")
        task_rows.append({**entry,
                          "success_rate": round(entry["successes"] / entry["episodes"], 3),
                          "mean_decisions_to_success": round(sum(d) / len(d), 2) if d else None,
                          "mean_tokens_per_decision": round(sum(t) / len(t)) if t else 0,
                          "cost_usd": round(entry["cost"], 5)})
    write_csv(args.out / "task_summary.csv", sorted(task_rows, key=lambda r: (r["task"], r["arm"])))

    methods = {}
    for r in task_rows:
        m = methods.setdefault(r["arm"], {"arm": r["arm"], "episodes": 0, "successes": 0,
                                          "censored": 0, "decisions": [], "tokens": [],
                                          "cost": 0.0})
        m["episodes"] += r["episodes"]
        m["successes"] += r["successes"]
        m["censored"] += r["censored"]
        if r["mean_decisions_to_success"]:
            m["decisions"].append(r["mean_decisions_to_success"])
        m["tokens"].append(r["mean_tokens_per_decision"])
        m["cost"] += r["cost_usd"]
    method_rows = []
    for m in methods.values():
        d, t = m.pop("decisions"), m.pop("tokens")
        method_rows.append({**m,
                            "success_rate": round(m["successes"] / m["episodes"], 3),
                            "mean_decisions_to_success": round(sum(d) / len(d), 2) if d else None,
                            "mean_tokens_per_decision": round(sum(t) / len(t)) if t else 0,
                            "cost_usd": round(m["cost"], 5)})
    write_csv(args.out / "method_summary.csv",
              sorted(method_rows, key=lambda r: r["arm"]))

    write_csv(args.out / "action_disagreement.csv", action_disagreement(args.out, rows))
    write_csv(args.out / "repeatability.csv", repeatability(args.out))

    total = sum(r["cost_usd"] for r in rows)
    (args.out / "cost_summary.json").write_text(json.dumps({
        "episodes": len(rows),
        "total_cost_usd": round(total, 5),
        "by_arm": {m["arm"]: m["cost_usd"] for m in method_rows},
    }, indent=2) + "\n")

    print("=== paired comparisons ===")
    for p in pairs:
        print(f"  {p['comparison']:20} {p['baseline']:16} -> {p['variant']:16} "
              f"blocks={p['paired_blocks']:2} both_solved={p['blocks_both_solved']:2} "
              f"faster={p['variant_faster']} slower={p['variant_slower']} "
              f"mean_delta={p['mean_decision_delta']}")
    print("\n=== per method ===")
    for m in sorted(method_rows, key=lambda r: r["arm"]):
        print(f"  {m['arm']:16} success {m['successes']}/{m['episodes']} "
              f"mean_dec={m['mean_decisions_to_success']} "
              f"tok/dec={m['mean_tokens_per_decision']} cost=${m['cost_usd']}")
    print(f"\nwrote pairs/task_summary/method_summary/action_disagreement/repeatability "
          f"under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
