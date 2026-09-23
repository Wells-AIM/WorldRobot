#!/usr/bin/env python3
"""Project Stage 2 pilot cost from measured Stage 1.5 token usage.

Character-based estimates ran about 3.5x low against reported tokens in Stage
1.5, so the projection here is anchored on *reported* provider tokens for the
arms that already ran, not on the estimator.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


TYPESAFE_USD_PER_MILLION = 0.042

# Measured in Stage 1.5 on top_drawer, reported input tokens per decision.
MEASURED = {
    "s1_original": {"tokens_per_decision": 1506, "decisions": 32, "source": "S1 arm"},
    "policy_compact": {"tokens_per_decision": 2701, "decisions": 26,
                       "source": "PolicyCompact arm"},
    "policy_shuffle": {"tokens_per_decision": 3316, "decisions": 33,
                       "source": "PolicyShuffle arm"},
}
# s1_compact has never run. It carries S1's facts under shorter keys with the
# legend sent once, so it should sit at or below S1; assumed equal to be safe.
ASSUMED = {
    "s1_compact": {"tokens_per_decision": 1506, "decisions": 32,
                   "source": "assumed equal to s1_original (compaction can only reduce)"},
}


def main(out=Path("results/stage2/cost_estimate.json")):
    rates = {**MEASURED, **ASSUMED}
    tasks = 3
    inits = 3
    pilot = {}
    total_tokens = 0
    for arm, spec in rates.items():
        episodes = tasks * inits
        # Decisions measured on top_drawer only; other tasks may differ, so the
        # projection uses the measured count as a per-episode expectation.
        tokens = episodes * spec["decisions"] * spec["tokens_per_decision"]
        total_tokens += tokens
        pilot[arm] = {
            "episodes": episodes,
            "expected_decisions_per_episode": spec["decisions"],
            "reported_tokens_per_decision": spec["tokens_per_decision"],
            "projected_input_tokens": tokens,
            "projected_cost_usd": round(tokens * TYPESAFE_USD_PER_MILLION / 1e6, 4),
            "basis": spec["source"],
        }

    repeat_spec = rates["policy_compact"]
    repeat_episodes = 9  # 3 tasks x 3 repeats of policy_compact
    repeat_tokens = repeat_episodes * repeat_spec["decisions"] * repeat_spec[
        "tokens_per_decision"]

    payload = {
        "generated_for": "Stage 2 pilot",
        "price_usd_per_million_input_tokens": TYPESAFE_USD_PER_MILLION,
        "basis": (
            "Reported provider tokens from the Stage 1.5 six-arm run on top_drawer. "
            "Decisions-per-episode are top_drawer figures used as an expectation for "
            "all three tasks; microwave and alphabet_soup may differ."
        ),
        "pilot_36_episodes": {
            "layout": f"{tasks} tasks x {inits} initial states x {len(rates)} arms",
            "per_arm": pilot,
            "projected_input_tokens": total_tokens,
            "projected_cost_usd": round(total_tokens * TYPESAFE_USD_PER_MILLION / 1e6, 4),
        },
        "repeatability_probe_9_episodes": {
            "layout": "3 tasks x policy_compact x 3 repeats",
            "projected_input_tokens": repeat_tokens,
            "projected_cost_usd": round(repeat_tokens * TYPESAFE_USD_PER_MILLION / 1e6, 4),
        },
        "projected_total_cost_usd": round(
            (total_tokens + repeat_tokens) * TYPESAFE_USD_PER_MILLION / 1e6, 4
        ),
        "runtime_note": (
            "policy_* arms simulate continuations inside the rollout: about 31 minutes "
            "of rollout per 60-decision episode measured in Stage 1.5. The S1 arms have "
            "no rollout cost. Expect roughly 6-9 hours of wall clock for 36 + 9 episodes."
        ),
        "caveats": [
            "Projection assumes top_drawer decision counts transfer to the other tasks.",
            "s1_compact has never run; its rate is assumed equal to s1_original.",
            "Per-episode budget guards cap actual spend regardless of this estimate.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("pilot_36_episodes", "repeatability_probe_9_episodes",
                       "projected_total_cost_usd")}, indent=2)[:1400])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
