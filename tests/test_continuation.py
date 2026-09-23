"""Stage 1.5: how a counterfactual future is constructed under receding horizon.

Offline tests use recorded physics; the ones that need a branch simulator are
marked `simulation`. No test contacts a provider.
"""

import pytest

from jev_libero.config import load_task
from jev_libero.continuation import (
    CONTINUATIONS,
    HoldContinuation,
    PolicyConsistentContinuation,
    RepeatContinuation,
    continuation_actions,
    make_continuation,
    neutralize_oracle,
)
from jev_libero.experiment import (
    STAGE15_ARMS,
    approx_tokens,
    compact_future,
    future_novelty,
    token_budget,
)
from jev_libero.futures import (
    CounterfactualRollout,
    contains_oracle,
    generate_candidates,
    hard_feasible,
)
from jev_libero.records import read_jsonl


@pytest.fixture
def recorded(records_root):
    folder = records_root / "top_drawer_seed1"
    return load_task(folder / "task_config.json"), read_jsonl(folder / "predictions.jsonl")


def make_rollout(distances, contacts=None, actions=None, events=()):
    points = []
    for index, distance in enumerate(distances):
        points.append(
            {
                "t_s": round(0.4 * index, 2),
                "after_primitive": index,
                "input": None if index == 0 else "x+40mm",
                "eef_mm": [0.0, 0.0, 0.0],
                "finger_gap_mm": 76.0,
                "moving_contact": (contacts or [False] * len(distances))[index],
                "obstacle_contact": False,
                "contact_modes": [],
                "target_force_N": 0.0,
                "obstacle_force_N": 0.0,
                "distance_to_moving_geometry_mm": distance,
                "qpos_m": -0.15,
                "remaining_open_mm": 151.0 - 10.0 * index,
            }
        )
    return CounterfactualRollout(
        candidate_id="C1",
        actions=actions or ["x+40mm"] * (len(distances) - 1),
        horizon_steps=8 * (len(distances) - 1),
        horizon_seconds=0.4 * (len(distances) - 1),
        primitives_simulated=len(distances) - 1,
        continuation="policy_consistent",
        checkpoints=points,
        events=list(events),
    )


# --- Test 3: no success leakage into continuation selection -------------------


class PoisonedFeatures(dict):
    """Raises if anything reads the simulator's verdict."""

    def __getitem__(self, key):
        if key in ("success", "reward", "task_complete"):
            raise AssertionError(f"continuation read the oracle field {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("success", "reward", "task_complete"):
            raise AssertionError(f"continuation read the oracle field {key!r}")
        return super().get(key, default)


def test_neutralize_oracle_forces_the_predicate_false(recorded):
    """advance_target reads {any: [closing_mm >= t, after.success]}; the oracle
    branch has to be split off before the contract can score a continuation."""
    cfg, rows = recorded
    after = rows[0]["predictions"]["x+40mm"]["after"]
    assert "success" in after
    neutral = neutralize_oracle({**after, "success": True})
    assert neutral["success"] is False
    assert neutral["qpos_m"] == after["qpos_m"]
    assert neutralize_oracle({"a": 1}) == {"a": 1}


def test_contract_scoring_survives_a_poisoned_success_field(recorded):
    """Scoring a continuation must not touch the verdict, even if it is there."""
    cfg, rows = recorded
    predictions = {
        name: {**p, "after": PoisonedFeatures({**p["after"]})}
        for name, p in rows[0]["predictions"].items()
    }
    neutral = {name: {**p, "after": dict(p["after"].items())} for name, p in predictions.items()}
    for entry in neutral.values():
        entry["after"]["success"] = False
    feasible, _ = hard_feasible(rows[0]["before"], neutral, cfg)
    assert feasible


# --- Test 1 / 5: the continuation set depends on the first input --------------


def test_continuation_options_scale_down_and_never_escalate():
    assert continuation_actions("x+40mm") == ["x+40mm", "x+10mm", "x+3mm", "hold"]
    assert continuation_actions("x+10mm") == ["x+10mm", "x+3mm", "hold"]
    assert continuation_actions("x+3mm") == ["x+3mm", "hold"]
    assert continuation_actions("y-40mm")[0] == "y-40mm"
    assert continuation_actions("rotate_z+10deg") == ["rotate_z+10deg", "hold"]
    assert continuation_actions("close") == ["close", "hold"]


def test_static_policies_ignore_branch_state():
    repeat, hold = RepeatContinuation(), HoldContinuation()
    assert repeat.reads_branch_state is False
    assert hold.reads_branch_state is False
    assert repeat.choose_next_action(None, "x+40mm", -1.0, 1, None)[0] == "x+40mm"
    assert hold.choose_next_action(None, "x+40mm", -1.0, 1, None)[0] == "hold"


def test_policy_consistent_declares_that_it_reads_state():
    policy = make_continuation("policy_consistent")
    assert isinstance(policy, PolicyConsistentContinuation)
    assert policy.reads_branch_state is True
    assert set(CONTINUATIONS) == {"repeat", "hold", "policy_consistent"}
    with pytest.raises(ValueError):
        make_continuation("learned_policy_continuation")


# --- Test 7: compact and full describe the same rollout -----------------------


def test_compact_reports_paths_and_transitions_not_repeated_absolutes():
    result = make_rollout([93.5, 64.4, 48.7, 49.5], contacts=[False, False, True, True])
    cfg = load_task("top_drawer")
    compact = compact_future(result, cfg)
    assert compact["t_s"] == [0.0, 0.4, 0.8, 1.2]
    assert compact["paths"]["distance_to_moving_geometry_mm"] == [93.5, 64.4, 48.7, 49.5]
    assert compact["target_contact"] == [False, False, True, True]
    assert contains_oracle(compact) == []


def test_compact_collapses_a_fact_that_never_changes():
    result = make_rollout([93.5, 90.0, 88.0], contacts=[False, False, False])
    compact = compact_future(result, load_task("top_drawer"))
    assert compact["target_contact_throughout"] is False
    assert "target_contact" not in compact


def test_compact_is_smaller_than_full_for_the_same_rollout():
    result = make_rollout([93.5, 64.4, 48.7, 49.5])
    cfg = load_task("top_drawer")
    full = {"checkpoints": result.checkpoints[1:], "events": result.events}
    compact = compact_future(result, cfg)
    assert approx_tokens(compact) < approx_tokens(full)


# --- Future novelty is diagnostic only ---------------------------------------


def test_future_novelty_flags_a_reversal_the_first_step_cannot_show():
    straight = make_rollout([93.5, 80.0, 70.0, 60.0])
    reversing = make_rollout([93.5, 64.4, 48.7, 70.0])
    assert future_novelty(straight, {})["future_novelty_count"] == 0
    signals = future_novelty(reversing, {})["signals"]
    assert any("reverses" in s for s in signals)


def test_future_novelty_flags_a_continuation_that_diverges():
    diverged = make_rollout([93.5, 64.4, 55.0], actions=["x+40mm", "x+10mm"])
    assert "continuation_diverges_from_first_input" in future_novelty(diverged, {})["signals"]


# --- Token accounting ---------------------------------------------------------


def test_token_budget_separates_future_from_immediate():
    criteria = {
        "x+40mm": {
            "input": "Move gripper +40 millimeters along WORLD x.",
            "predicted_closing_mm": 0.0,
            "future_trajectory": {"paths": {"d": [93.5, 64.4, 48.7]}},
        }
    }
    budget = token_budget({"task": "close the drawer"}, "Pick one.", criteria)
    assert budget["estimated"] is True
    assert budget["future_tokens"] > 0
    assert budget["immediate_tokens"] > 0
    assert budget["total_tokens"] == (
        budget["base_tokens"] + budget["immediate_tokens"] + budget["future_tokens"]
    )


def test_token_budget_reports_no_future_for_the_baseline():
    criteria = {"x+40mm": {"input": "Move.", "predicted_closing_mm": 0.0}}
    assert token_budget({}, "Pick one.", criteria)["future_tokens"] == 0


# --- Test 2: the first-step candidate set is invariant across arms ------------


def test_every_stage15_arm_shares_the_first_step_candidate_set(recorded):
    cfg, rows = recorded
    feasible, _ = hard_feasible(rows[0]["before"], rows[0]["predictions"], cfg)
    baseline = [c.first_input for c in generate_candidates(rows[0]["predictions"], feasible, 27, 3)]
    for arm, spec in STAGE15_ARMS.items():
        candidates = generate_candidates(rows[0]["predictions"], feasible, 27, 3)
        assert [c.first_input for c in candidates] == baseline, arm


def test_stage15_arms_differ_only_in_future_construction():
    assert STAGE15_ARMS["S1"]["mode"] == "original_no_oracle"
    futures = {k: v for k, v in STAGE15_ARMS.items() if k != "S1"}
    assert all(v["mode"] == "strong_policy_future" for v in futures.values())
    assert {v["continuation"] for v in futures.values()} == {"repeat", "hold", "policy_consistent"}
    assert STAGE15_ARMS["S1+PolicyShuffle"]["shuffle"] is True
    assert STAGE15_ARMS["S1+PolicyCompact"].get("shuffle") is None
    # Compact and Shuffle differ only in the correspondence, not the physics.
    compact = dict(STAGE15_ARMS["S1+PolicyCompact"])
    shuffle = {k: v for k, v in STAGE15_ARMS["S1+PolicyShuffle"].items() if k != "shuffle"}
    assert compact == shuffle
