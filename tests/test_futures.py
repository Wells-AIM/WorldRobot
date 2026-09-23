"""Offline controlled-experiment tests. Never contacts a provider or a simulator.

Recorded predictions from examples/records supply real physics data, so
candidate generation and prompt construction are exercised without LIBERO.
"""

from collections import Counter

import pytest

from jev_libero.config import load_task
from jev_libero.experiment import (
    CONTROLLED_MODES,
    MODES,
    ORIGINAL_MODES,
    OracleFilteringAPI,
    decide,
    horizon_for,
    prune_oracle,
)
from jev_libero.futures import (
    CONTINUATIONS,
    ORACLE_MARKERS,
    allocate,
    chunk_actions,
    contains_oracle,
    derangement,
    generate_candidates,
    hard_feasible,
    spread,
    strip_oracle,
)
from jev_libero.policy import family
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


def test_modes_include_the_original_variants_and_four_controlled_arms():
    assert MODES == (*ORIGINAL_MODES, *CONTROLLED_MODES)
    assert ORIGINAL_MODES == (
        "original",
        "original_no_oracle",
        "original_deep",
        "original_trajectory",
        "strong_policy_future",
    )
    assert CONTROLLED_MODES == (
        "reactive",
        "short_preview",
        "counterfactual_future",
        "shuffle_future",
    )


def test_trajectory_option_keeps_the_executable_step_visible(recorded):
    """original_trajectory adds the path without replacing the executed step.

    This is the distinction original_deep got wrong: it swapped the one-primitive
    derived fields for a chunk endpoint, so the critic optimised a chunk that the
    receding horizon never finishes.
    """
    from jev_libero.policy import ValidatedPolicy

    cfg, rows = recorded
    row = rows[0]
    predictions = {
        name: {
            **p,
            "trajectory": {
                "horizon_s": 1.2,
                "checkpoints": [
                    {"t_s": 0.4, "eef_mm": [0, 0, 0]},
                    {"t_s": 0.8, "eef_mm": [0, 0, 0]},
                    {"t_s": 1.2, "eef_mm": [0, 0, 0]},
                ],
                "events": [],
            },
        }
        for name, p in row["predictions"].items()
    }
    api = MockJev()
    ValidatedPolicy(True, cfg).choose(api, 0, row["before"], predictions)
    motor = next(r for r in api.requests if r["layer"] == "motor")
    name, option = next(iter(motor["criteria"].items()))
    assert "future_trajectory" in option
    assert [c["t_s"] for c in option["future_trajectory"]["checkpoints"]] == [0.4, 0.8, 1.2]
    # The one-primitive fields the published pipeline relies on are still there.
    assert option["predicted_closing_mm"] == round(row["predictions"][name]["closing_mm"], 3)
    assert "predicted_surface_gap_reduction_mm" in option


def test_original_prompt_has_no_trajectory_field(recorded):
    """Without the field, the published pipeline's prompt is untouched."""
    from jev_libero.policy import ValidatedPolicy

    cfg, rows = recorded
    api = MockJev()
    ValidatedPolicy(True, cfg).choose(api, 0, rows[0]["before"], rows[0]["predictions"])
    motor = next(r for r in api.requests if r["layer"] == "motor")
    assert all("future_trajectory" not in o for o in motor["criteria"].values())


def test_prune_oracle_keeps_the_task_aligned_fields(recorded):
    """The no-oracle variant must lose the verdict but keep the physics."""
    option = {
        "input": "Move gripper +3 millimeters along WORLD x; hold orientation.",
        "predicted_task_complete": False,
        "predicted_closing_mm": 0.0,
        "predicted_surface_gap_reduction_mm": 2.42,
        "predicted_target_contact": False,
        "predicted_obstacle_peak_force_N": 0.0,
    }
    pruned = prune_oracle(option)
    assert "predicted_task_complete" not in pruned
    assert pruned["predicted_surface_gap_reduction_mm"] == 2.42
    assert pruned["predicted_target_contact"] is False
    assert contains_oracle(pruned) == []
    nested = {"a": {"can_finish_task": True, "keep": 1}, "b": [{"terminal_success": 1, "x": 2}]}
    assert prune_oracle(nested) == {"a": {"keep": 1}, "b": [{"x": 2}]}


def test_oracle_filtering_api_strips_real_recorded_requests(records_root):
    """Replay the published prompts through the filter and check what survives."""
    calls = read_jsonl(records_root / "top_drawer_seed1" / "api.jsonl")
    leaky = [c for c in calls if contains_oracle(c["request"]["questions"][c["layer"]]["criteria"])]
    assert leaky, "the published pipeline should expose the success predicate"
    inner = MockJev()
    api = OracleFilteringAPI(inner)
    for call in calls:
        layer = call["layer"]
        question = call["request"]["questions"][layer]
        api.choose(
            call["step"],
            layer,
            call["request"]["state"],
            question["instructions"],
            question["criteria"],
        )
    for request in inner.requests:
        assert contains_oracle(request["state"]) == []
        assert contains_oracle(request["criteria"]) == []
    assert api.calls == len(calls)


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


def test_allocation_is_proportional_not_round_robin(recorded):
    """Regression: round-robin gave a menu that excluded every approach move.

    At the top_drawer start state the feasible pool is 18 translations, 6
    rotations, 2 finger inputs and 1 hold. Round-robin handed Jev one single
    translation out of six candidates; proportional allocation keeps the menu
    shaped like the pool.
    """
    cfg, rows = recorded
    feasible, _ = hard_feasible(rows[0]["before"], rows[0]["predictions"], cfg)
    sizes = Counter(
        family(name, rows[0]["predictions"][name]).split("__")[1] for name in feasible
    )
    assert sizes["translation"] > sizes["rotation"] > sizes["finger"]
    candidates = generate_candidates(rows[0]["predictions"], feasible, 6, 3)
    kinds = Counter(c.family.split("__")[1] for c in candidates)
    assert kinds["translation"] >= 4, kinds
    assert kinds["translation"] > kinds["rotation"], kinds


def test_allocate_respects_totals_and_capacity():
    sizes = {"translation": 18, "rotation": 6, "finger": 2, "hold": 1}
    for count in range(1, 28):
        slots = allocate(sizes, count)
        assert sum(slots.values()) == count, (count, slots)
        assert all(slots[key] <= sizes[key] for key in sizes), (count, slots)
        assert all(value >= 0 for value in slots.values()), (count, slots)
    assert allocate(sizes, 27) == sizes


def test_spread_covers_both_endpoints():
    assert spread(list(range(18)), 4) == [0, 6, 11, 17]
    assert spread([1, 2, 3], 5) == [1, 2, 3]
    assert spread([1, 2, 3], 1) == [1]


def test_larger_candidate_budgets_cover_more_of_the_action_space(recorded):
    """K=6 cannot cover 27 inputs; the menu must widen as K grows."""
    cfg, rows = recorded
    feasible, _ = hard_feasible(rows[0]["before"], rows[0]["predictions"], cfg)
    seen = {}
    for count in (6, 12, 27):
        names = {c.first_input for c in generate_candidates(rows[0]["predictions"], feasible, count, 1)}
        seen[count] = names
        assert len(names) == min(count, len(feasible))
    assert seen[6] < seen[12] or len(seen[6] - seen[12]) <= 2
    assert seen[27] == set(feasible)


def test_oracle_markers_cover_the_original_leaky_fields():
    """The original prompt exposes these; the controlled arms must not."""
    for field in ("success", "can_finish_task", "predicted_task_complete", "terminal_success"):
        assert contains_oracle({field: True}) == [field], field
    assert "success" in ORACLE_MARKERS


def test_chunk_continuation_shapes_the_sequence():
    assert chunk_actions("x+40mm", 3, "repeat") == ["x+40mm"] * 3
    assert chunk_actions("x+40mm", 3, "hold") == ["x+40mm", "hold", "hold"]
    assert chunk_actions("x+40mm", 1, "hold") == ["x+40mm"]
    assert set(CONTINUATIONS) == {"repeat", "hold", "policy_consistent"}
    with pytest.raises(ValueError):
        chunk_actions("x+40mm", 3, "search")


def test_continuation_reaches_the_candidate_set(recorded):
    """The chunk the critic is shown must follow the requested continuation."""
    cfg, rows = recorded
    feasible, _ = hard_feasible(rows[0]["before"], rows[0]["predictions"], cfg)
    repeat = generate_candidates(rows[0]["predictions"], feasible, 6, 3, "repeat")
    hold = generate_candidates(rows[0]["predictions"], feasible, 6, 3, "hold")
    assert [c.first_input for c in repeat] == [c.first_input for c in hold]
    assert all(len(set(c.actions)) == 1 for c in repeat)
    assert all(c.actions[1:] == ["hold", "hold"] for c in hold)
