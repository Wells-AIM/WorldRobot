"""Offline controlled-experiment tests. Never contacts a provider or a simulator.

Recorded predictions from examples/records supply real physics data, so
candidate generation and prompt construction are exercised without LIBERO.
"""

import pytest

from jev_libero.config import load_task
from jev_libero.experiment import CONTROLLED_MODES, MODES, decide, horizon_for
from jev_libero.futures import (
    ORACLE_MARKERS,
    contains_oracle,
    derangement,
    generate_candidates,
    hard_feasible,
    strip_oracle,
)
from jev_libero.records import read_jsonl


class MockJev:
    """Deterministic offline stand-in for Jev. MOCK OUTPUT, not model behaviour.

    It always takes the first listed candidate, so any difference between two
    modes comes from the pipeline, never from a sampled answer.
    """

    def __init__(self):
        self.total = 0.0
        self.calls = 0
        self.requests = []

    def choose(self, step, layer, state, instructions, criteria):
        self.calls += 1
        self.requests.append(
            {
                "step": step,
                "layer": layer,
                "state": state,
                "instructions": instructions,
                "criteria": criteria,
                "mock": True,
            }
        )
        return next(iter(criteria))

    def close(self):
        pass


@pytest.fixture
def recorded(records_root):
    folder = records_root / "top_drawer_seed1"
    cfg = load_task(folder / "task_config.json")
    rows = read_jsonl(folder / "predictions.jsonl")
    return cfg, rows


def test_derangement_has_no_fixed_point():
    ids = [f"C{i}" for i in range(1, 7)]
    mapping = derangement(ids, seed=3)
    assert sorted(mapping) == sorted(ids)
    assert sorted(mapping.values()) == sorted(ids)
    assert all(key != value for key, value in mapping.items())


def test_derangement_is_deterministic():
    ids = [f"C{i}" for i in range(1, 7)]
    assert derangement(ids, seed=11) == derangement(ids, seed=11)
    assert derangement(ids, seed=11) != derangement(ids, seed=12)


def test_derangement_rejects_single_candidate():
    with pytest.raises(ValueError):
        derangement(["C1"], seed=0)


def test_strip_oracle_drops_the_success_predicate(recorded):
    cfg, rows = recorded
    before = rows[0]["before"]
    assert before["success"] is False
    clean = strip_oracle(before, cfg)
    assert "success" not in clean
    assert clean["eef_mm"] == before["eef_mm"]
    assert clean["qpos_m"] == before["qpos_m"]


def test_contains_oracle_finds_planted_markers():
    assert contains_oracle({"a": {"success": True}}) == ["a.success"]
    assert contains_oracle({"note": "this is the best_action"}) == ["note"]
    assert contains_oracle([{"task_score": 1}]) == ["[0].task_score"]
    assert contains_oracle({"eef_mm": [1, 2, 3], "finger_gap_mm": 4.0}) == []


def test_candidate_set_is_identical_across_controlled_modes(recorded):
    """Candidate fairness: same task, same state, same seed, same candidates."""
    cfg, rows = recorded
    for row in rows[:6]:
        feasible, _ = hard_feasible(row["before"], row["predictions"], cfg)
        if not feasible:
            continue
        sets = []
        for _ in CONTROLLED_MODES:
            candidates = generate_candidates(row["predictions"], feasible, 6, 3)
            sets.append([(c.candidate_id, tuple(c.actions), c.family) for c in candidates])
        assert all(entry == sets[0] for entry in sets)


def test_candidate_generation_ignores_the_success_predicate(recorded):
    """Flipping the oracle verdict must not move a single candidate."""
    cfg, rows = recorded
    row = rows[0]
    feasible, _ = hard_feasible(row["before"], row["predictions"], cfg)
    baseline = generate_candidates(row["predictions"], feasible, 6, 3)
    flipped = {
        name: {**p, "after": {**p["after"], "success": not p["after"]["success"]}}
        for name, p in row["predictions"].items()
    }
    feasible_flipped, _ = hard_feasible(row["before"], flipped, cfg)
    after = generate_candidates(flipped, feasible_flipped, 6, 3)
    assert [c.as_record() for c in baseline] == [c.as_record() for c in after]


def test_candidate_depth_and_count_are_configurable(recorded):
    cfg, rows = recorded
    feasible, _ = hard_feasible(rows[0]["before"], rows[0]["predictions"], cfg)
    for count in (2, 4, 6):
        for depth in (1, 2, 3):
            candidates = generate_candidates(rows[0]["predictions"], feasible, count, depth)
            assert len(candidates) <= count
            assert all(len(c.actions) == depth for c in candidates)


def test_hard_rejections_partition_the_candidate_pool(recorded):
    cfg, rows = recorded
    for row in rows:
        feasible, rejected = hard_feasible(row["before"], row["predictions"], cfg)
        assert set(feasible) | set(rejected) == set(row["predictions"])
        assert not set(feasible) & set(rejected)
    rejecting = [
        row for row in rows if hard_feasible(row["before"], row["predictions"], cfg)[1]
    ]
    assert rejecting, "expected at least one state where the collision rules reject an input"


def test_horizon_mapping():
    assert horizon_for("reactive", 1.2) == 0.0
    assert horizon_for("short_preview", 1.2) == 0.4
    for horizon in (0.0, 0.4, 0.8, 1.2, 2.0):
        assert horizon_for("counterfactual_future", horizon) == horizon
        assert horizon_for("shuffle_future", horizon) == horizon


def test_modes_include_original_and_four_controlled_arms():
    assert MODES == ("original", *CONTROLLED_MODES)
    assert CONTROLLED_MODES == (
        "reactive",
        "short_preview",
        "counterfactual_future",
        "shuffle_future",
    )


def test_reactive_prompt_carries_no_future_and_no_oracle(recorded):
    """Mode 1 sees the task, the current state, and the candidate list only."""
    cfg, rows = recorded
    api = MockJev()
    choice, record = decide(
        api, None, 0, "reactive", rows[0]["before"], rows[0]["predictions"], -1.0, cfg
    )
    request = api.requests[0]
    assert all("future" not in option for option in request["criteria"].values())
    assert all("planned_chunk" not in option for option in request["criteria"].values())
    assert record["future_horizon_requested_s"] == 0.0
    assert record["candidates"][0]["counterfactual"] is None
    assert contains_oracle(request["state"]) == []
    assert contains_oracle(request["criteria"]) == []
    assert choice == record["executed_input"]


def test_no_reward_leakage_in_any_controlled_prompt(recorded):
    """Test 6: nothing Jev reads may carry the simulator's own verdict."""
    cfg, rows = recorded
    for mode in CONTROLLED_MODES:
        api = MockJev()
        if mode == "reactive":
            decide(api, None, 0, mode, rows[0]["before"], rows[0]["predictions"], -1.0, cfg)
        else:
            continue  # simulated modes are covered in the simulation suite
        for request in api.requests:
            assert contains_oracle(request["state"]) == []
            assert contains_oracle(request["criteria"]) == []
            assert contains_oracle(request["instructions"]) == []


def test_recent_outcomes_are_stripped_of_the_success_predicate(recorded):
    """Regression: the task's feedback projection leaked `success` into the state.

    top_drawer declares a `success` feedback field, so a non-empty history put
    the LIBERO predicate in front of the critic on every decision after the
    first. Only a multi-step check catches this; step 0 has no history.
    """
    cfg, rows = recorded
    history = [
        {"input": "y-40mm", "actual_closing_mm": 1.2, "success": False},
        {"input": "y-40mm", "actual_closing_mm": 2.4, "success": True},
    ]
    api = MockJev()
    decide(
        api,
        None,
        3,
        "reactive",
        rows[0]["before"],
        rows[0]["predictions"],
        -1.0,
        cfg,
        history=history,
    )
    state = api.requests[0]["state"]
    assert state["recent_real_outcomes"], "history must still reach the critic"
    assert all("success" not in entry for entry in state["recent_real_outcomes"])
    assert all("actual_closing_mm" in entry for entry in state["recent_real_outcomes"])
    assert contains_oracle(state) == []


def test_oracle_markers_cover_the_original_leaky_fields():
    """The original prompt exposes these; the controlled arms must not."""
    for field in ("success", "can_finish_task", "predicted_task_complete", "terminal_success"):
        assert contains_oracle({field: True}) == [field], field
    assert "success" in ORACLE_MARKERS
