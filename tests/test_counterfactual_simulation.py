"""Counterfactual-future tests that need LIBERO. Offline: no provider is called."""

import json

import numpy as np
import pytest
from test_futures import MockJev

from jev_libero.experiment import CONTROLLED_MODES, decide
from jev_libero.futures import (
    contains_oracle,
    generate_candidates,
    hard_feasible,
    rollout,
    rollout_all,
)

pytestmark = pytest.mark.simulation

HORIZON = 1.2
DEPTH = 3
COUNT = 6


def live_state(world):
    """Everything a counterfactual must leave untouched."""
    return {
        "qpos": world.data.qpos.copy(),
        "qvel": world.data.qvel.copy(),
        "time": float(world.data.time),
        "sim_state": world.env.sim.get_state().flatten().copy(),
    }


def assert_same_state(a, b, atol=0.0):
    np.testing.assert_allclose(a["qpos"], b["qpos"], atol=atol, rtol=0)
    np.testing.assert_allclose(a["qvel"], b["qvel"], atol=atol, rtol=0)
    np.testing.assert_allclose(a["sim_state"], b["sim_state"], atol=atol, rtol=0)
    assert a["time"] == pytest.approx(b["time"], abs=max(atol, 1e-12))


@pytest.fixture
def world_with_candidates(simulator):
    world = simulator(seed=1, task_config="top_drawer")
    try:
        before, predictions = world.predict_all(-1.0)
        feasible, _ = hard_feasible(before, predictions, world.config)
        candidates = generate_candidates(predictions, feasible, COUNT, DEPTH)
        yield world, before, predictions, candidates
    finally:
        world.close()


def test_rollout_does_not_contaminate_the_live_environment(world_with_candidates):
    """Test 1: simulator isolation."""
    world, _, _, candidates = world_with_candidates
    baseline = live_state(world)
    for candidate in candidates:
        rollout(world, candidate, -1.0, HORIZON, world.config)
        assert_same_state(baseline, live_state(world))


def test_every_candidate_rolls_out_from_the_same_snapshot(world_with_candidates):
    """Test 1 continued: C2's rollout must start where C1's started, not where it ended."""
    world, _, _, candidates = world_with_candidates
    baseline = live_state(world)
    rollouts = rollout_all(world, candidates, -1.0, HORIZON, world.config)
    assert_same_state(baseline, live_state(world))
    assert len(rollouts) == len(candidates)
    for result in rollouts.values():
        start = result.checkpoints[0]
        np.testing.assert_allclose(
            start["eef_mm"], (world.obs["robot0_eef_pos"] * 1000).tolist(), atol=1e-9, rtol=0
        )


def test_rollout_is_deterministic(world_with_candidates):
    """Test 3: same snapshot, same candidate, same trajectory."""
    world, _, _, candidates = world_with_candidates
    first = rollout_all(world, candidates, -1.0, HORIZON, world.config)
    second = rollout_all(world, candidates, -1.0, HORIZON, world.config)
    for key in first:
        assert first[key].checkpoints == second[key].checkpoints
        assert first[key].events == second[key].events
        assert first[key].summary == second[key].summary
        assert first[key].horizon_steps == second[key].horizon_steps


def test_predicted_future_matches_real_execution(world_with_candidates):
    """Test 4: the oracle rollout is the same physics the live env will run."""
    world, _, _, candidates = world_with_candidates
    from jev_libero.world import Snapshot

    candidate = candidates[0]
    predicted = rollout(world, candidate, -1.0, HORIZON, world.config)
    snapshot = Snapshot(world)
    try:
        grip = -1.0
        for name in candidate.actions[: predicted.primitives_simulated]:
            result = world.execute(name, grip)
            grip = result["grip"]
        actual = world.features()
        final = predicted.checkpoints[-1]
        np.testing.assert_allclose(final["eef_mm"], actual["eef_mm"], atol=1e-9, rtol=0)
        assert final["finger_gap_mm"] == pytest.approx(actual["finger_gap_mm"], abs=1e-9)
        assert final["distance_to_moving_geometry_mm"] == pytest.approx(
            actual["distance_to_moving_geometry_mm"], abs=1e-9
        )
    finally:
        snapshot.restore()


def test_horizon_controls_how_far_the_future_reaches(world_with_candidates):
    """The horizon knob the ablation will sweep actually changes simulated depth."""
    world, _, _, candidates = world_with_candidates
    candidate = candidates[0]
    depths = {}
    for horizon in (0.4, 0.8, 1.2):
        result = rollout(world, candidate, -1.0, horizon, world.config)
        depths[horizon] = result.primitives_simulated
        assert result.horizon_seconds <= horizon + 1e-6
        assert len(result.checkpoints) == result.primitives_simulated + 1
    assert depths[0.4] == 1
    assert depths[0.8] == 2
    assert depths[1.2] == 3


def test_candidate_sets_match_across_modes_on_live_physics(world_with_candidates):
    """Test 2: candidate fairness, measured against the real simulator."""
    world, before, predictions, _ = world_with_candidates
    sets = {}
    for mode in CONTROLLED_MODES:
        api = MockJev()
        _, record = decide(
            api,
            world,
            0,
            mode,
            before,
            predictions,
            -1.0,
            world.config,
            candidate_count=COUNT,
            candidate_depth=DEPTH,
            future_horizon_s=HORIZON,
        )
        sets[mode] = [(c["candidate_id"], tuple(c["actions"])) for c in record["candidates"]]
    assert len({tuple(value) for value in sets.values()}) == 1


def test_simulated_prompts_carry_no_oracle_verdict(world_with_candidates):
    """Test 6 against real rollouts, including the task's own success feature."""
    world, before, predictions, _ = world_with_candidates
    for mode in CONTROLLED_MODES:
        api = MockJev()
        decide(
            api,
            world,
            0,
            mode,
            before,
            predictions,
            -1.0,
            world.config,
            candidate_count=COUNT,
            candidate_depth=DEPTH,
            future_horizon_s=HORIZON,
        )
        request = api.requests[0]
        assert contains_oracle(request["state"]) == [], mode
        assert contains_oracle(request["criteria"]) == [], mode
        assert contains_oracle(request["instructions"]) == [], mode


def test_shuffle_future_deranges_the_correspondence(world_with_candidates):
    """Test 5: no candidate may receive its own future."""
    world, before, predictions, _ = world_with_candidates
    api = MockJev()
    _, record = decide(
        api,
        world,
        0,
        "shuffle_future",
        before,
        predictions,
        -1.0,
        world.config,
        candidate_count=COUNT,
        candidate_depth=DEPTH,
        future_horizon_s=HORIZON,
        shuffle_seed=7,
    )
    pairs = [(c["original_future_id"], c["assigned_future_id"]) for c in record["candidates"]]
    assert all(original != assigned for original, assigned in pairs)
    assert sorted(a for _, a in pairs) == sorted(o for o, _ in pairs)


def test_shuffle_keeps_the_same_future_payload_shape(world_with_candidates):
    """Only the correspondence changes; field names and counts stay put."""
    world, before, predictions, _ = world_with_candidates
    shapes = {}
    for mode in ("counterfactual_future", "shuffle_future"):
        api = MockJev()
        decide(
            api,
            world,
            0,
            mode,
            before,
            predictions,
            -1.0,
            world.config,
            candidate_count=COUNT,
            candidate_depth=DEPTH,
            future_horizon_s=HORIZON,
            shuffle_seed=7,
        )
        criteria = api.requests[0]["criteria"]
        shapes[mode] = sorted(
            (name, tuple(sorted(option["future"])), len(option["future"]["checkpoints"]))
            for name, option in criteria.items()
        )
    assert shapes["counterfactual_future"] == shapes["shuffle_future"]


@pytest.mark.parametrize("mode", CONTROLLED_MODES)
def test_mock_episode_runs_end_to_end(simulator, tmp_path, monkeypatch, mode):
    """Success criterion H: every controlled arm completes a full pipeline."""
    import jev_libero.runner as runner

    monkeypatch.setattr(runner, "Decisions", lambda *a, **k: MockJev())
    out = tmp_path / mode
    result = runner.run(
        "top_drawer",
        out,
        seed=1,
        render=False,
        mode=mode,
        max_decisions=3,
        candidate_count=COUNT,
        candidate_depth=DEPTH,
        future_horizon_s=HORIZON,
    )
    assert result["experiment_mode"] == mode
    assert 1 <= result["decisions"] <= 3
    rows = [json.loads(line) for line in (out / "counterfactual.jsonl").read_text().splitlines()]
    assert len(rows) == result["decisions"]
    assert all(row["candidate_depth"] == DEPTH for row in rows)
    assert all(row["experiment_mode"] == mode for row in rows)


def test_receding_horizon_executes_only_the_first_primitive(simulator, tmp_path, monkeypatch):
    """Test 7: a depth-3 chunk still advances the live env by one primitive."""
    import jev_libero.runner as runner

    monkeypatch.setattr(runner, "Decisions", lambda *a, **k: MockJev())
    out = tmp_path / "receding"
    result = runner.run(
        "top_drawer",
        out,
        seed=1,
        render=False,
        mode="counterfactual_future",
        max_decisions=4,
        candidate_count=COUNT,
        candidate_depth=DEPTH,
        future_horizon_s=HORIZON,
    )
    rows = [json.loads(line) for line in (out / "counterfactual.jsonl").read_text().splitlines()]
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    for row, step in zip(rows, trace):
        chosen = next(
            c for c in row["candidates"] if c["candidate_id"] == row["selected_candidate"]
        )
        assert len(chosen["actions"]) == DEPTH
        assert step["choice"] == chosen["actions"][0] == row["executed_input"]
    # Each decision advances the live env by at most one 8-step primitive,
    # however deep the candidate chunk that was reasoned over.
    assert result["decisions"] == len(rows)
    assert result["sim_steps"] <= result["decisions"] * 8


def test_original_mode_is_untouched(simulator, records_root, tmp_path, monkeypatch):
    """Test 8: the published pipeline still reproduces its recorded episode."""
    import jev_libero.runner as runner
    from jev_libero.records import read_jsonl

    folder = records_root / "top_drawer_seed1"
    calls = read_jsonl(folder / "api.jsonl")

    class OfflineAPI:
        def __init__(self, *args, **kwargs):
            self.total = 0.0
            self.calls = 0

        def choose(self, step, layer, state, instructions, criteria):
            c = calls[self.calls]
            self.calls += 1
            assert (step, layer) == (c["step"], c["layer"])
            assert state == c["request"]["state"]
            assert criteria == c["request"]["questions"][layer]["criteria"]
            self.total += c["response"]["usage"]["cost"]
            return c["response"]["answers"][layer]["choice"]

        def close(self):
            pass

    monkeypatch.setattr(runner, "Decisions", OfflineAPI)
    out = tmp_path / "original"
    result = runner.run("top_drawer", out, seed=1, render=False)
    assert result["success"] and result["decisions"] == 20 and result["sim_steps"] == 155
    assert result["experiment_mode"] == "original"
    assert not (out / "counterfactual.jsonl").exists()
    # Every recorded request was replayed and accepted verbatim, so the prompts,
    # the routing, and the chosen inputs are unchanged by the experiment modes.
    lines = (out / "trace.jsonl").read_text().splitlines()
    executed = [json.loads(line)["choice"] for line in lines]
    assert executed == [
        row["response"]["answers"]["motor"]["choice"] for row in calls if row["layer"] == "motor"
    ]
    # Controls match the published record to simulator tolerance. Exact equality
    # is a hardware-determinism property this host does not reproduce; the
    # upstream test test_published_controls_replay fails here identically on an
    # unmodified checkout, at ~6e-12.
    np.testing.assert_allclose(
        np.load(out / "sim_trajectory.npz")["actions"],
        np.load(folder / "sim_trajectory.npz")["actions"],
        atol=1e-6,
        rtol=0,
    )
