#!/usr/bin/env python3
"""Phase analysis for the Stage 1.5 arms: where the decisions go, and on what."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_libero.records import read_jsonl  # noqa: E402

ORDER = ["S1", "S1_Repeat", "S1_Hold", "S1_PolicyFull", "S1_PolicyCompact", "S1_PolicyShuffle"]
PHASES = ["far", "mid", "near", "terminal"]


def size(name):
    m = re.match(r"[xyz]([+-]\d+)mm", name)
    return abs(int(m.group(1))) if m else None


def main(base="runs/stage15_real"):
    base = Path(base)
    arms = [a for a in ORDER if (base / a / "summary.json").exists()]

    print("=== outcome ===")
    print("%-18s%9s%6s%10s%12s%22s" % ("arm", "success", "dec", "final mm", "cost", "termination"))
    data = {}
    for arm in arms:
        s = json.loads((base / arm / "summary.json").read_text())
        final = json.loads((base / arm / "final_state.json").read_text())
        data[arm] = (s, read_jsonl(base / arm / "trace.jsonl"))
        print("%-18s%9s%6d%10.2f%12s%22s" % (
            arm, s["success"], s["decisions"], final["remaining_open_mm"],
            "$%.5f" % s["cost_usd"], s["termination"]))

    print("\n=== decisions by phase (far >50, mid 20-50, near 5-20, terminal <5 mm) ===")
    print("%-18s%8s%8s%8s%10s" % ("arm", *PHASES))
    for arm in arms:
        s, _ = data[arm]
        p = s.get("decisions_by_phase", {})
        print("%-18s%8d%8d%8d%10d" % (arm, *(p.get(k, 0) for k in PHASES)))

    print("\n=== mean progress mm/decision, by phase ===")
    print("%-18s%8s%8s%8s%10s" % ("arm", *PHASES))
    for arm in arms:
        _, trace = data[arm]
        cells = []
        for phase in PHASES:
            rows = [r for r in trace if r.get("phase") == phase]
            gained = sum(r["before_task_value"] - r["after_task_value"] for r in rows)
            cells.append(gained / len(rows) if rows else 0.0)
        print("%-18s%8.2f%8.2f%8.2f%10.2f" % (arm, *cells))

    print("\n=== action size mix, by phase (40/10/3 mm) ===")
    for arm in arms:
        _, trace = data[arm]
        parts = []
        for phase in PHASES:
            rows = [r for r in trace if r.get("phase") == phase]
            counts = {40: 0, 10: 0, 3: 0}
            for r in rows:
                v = size(r["choice"])
                if v in counts:
                    counts[v] += 1
            parts.append("%s %d/%d/%d" % (phase, counts[40], counts[10], counts[3]))
        print("  %-18s %s" % (arm, "   ".join(parts)))

    print("\n=== estimated prompt split per decision ===")
    print("%-18s%10s%12s%10s%12s" % ("arm", "base", "immediate", "future", "total"))
    for arm in arms:
        _, trace = data[arm]
        keys = ("base_tokens", "immediate_tokens", "future_tokens", "total_tokens")
        sums = {k: 0 for k in keys}
        for r in trace:
            b = r.get("token_budget_estimated", {})
            for k in keys:
                sums[k] += b.get(k, 0)
        n = max(len(trace), 1)
        print("%-18s%10d%12d%10d%12d" % (arm, *(round(sums[k] / n) for k in keys)))

    print("\n=== reported provider tokens per decision ===")
    for arm in arms:
        s, trace = data[arm]
        api = read_jsonl(base / arm / "api.jsonl")
        got = sum(c.get("response", {}).get("usage", {}).get("input_tokens", 0) for c in api)
        print("  %-18s %6d  (%d calls)" % (arm, round(got / max(len(trace), 1)), len(api)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/stage15_real")
