"""Stage 1.5 tests that need a branch simulator. Offline: no provider is called."""

import json

import numpy as np
import pytest
from test_futures import MockJev

from jev_libero.continuation import make_continuation
from jev_libero.experiment import attach_trajectories
from jev_libero.futures import CandidateSequence, contains_oracle, hard_feasible, rollout

pytestmark = pytest.mark.simulation

HORIZON = 1.2
DEPTH = 3


@pytest.fixture
def world(simulator):
    w = simulator(seed=1, task_config="top_drawer")
    try:
        yield w
    finally:
        w.close()


def chunk(first, depth=DEPTH):
    return CandidateSequence(
        candidate_id=first, actions=[first] * depth, family="", first_input=first
    )


# --- Test 1 + 5: the continuation reacts to the branch state ------------------


def test_policy_continuation_diverges_from_repeat_across_the_candidate_set(world):
    """It must not be a disguised repeat.

    Not every candidate has to diverge: when continuing genuinely is the best
    available option the policy should say so, and at the start state `x+40mm`
    is exactly that case. What has to hold is that the policy re-decides, so
    across a real candidate set some chunks stop looking like `[a, a, a]`.
    """
    policy = make_continuation("policy_consistent")
    before, predictions = world.predict_all(-1.0)
    feasible, _ = hard_feasible(before, predictions, world.config)
    names = [n for n in feasible if n not in ("hold", "open", "close")][:10]
    diverged = {}
    for first in names:
        result = rollout(world, chunk(first), -1.0, HORIZON, world.config,
                         continuation=policy, depth=DEPTH)
        assert result.actions[0] == first
        assert len(result.actions) == DEPTH
        assert all(r["rule"] for r in result.continuation_reasons)
        if len(set(result.actions)) > 1:
            diverged[first] = result.actions
    assert diverged, f"policy continuation behaved as repeat for every one of {names}"


def test_policy_continuation_stops_a_retreat_instead_of_repeating_it(world):
    """A first input that moves away from the target should not be shown as
    three of itself; that is the misrepresentation `repeat` cannot avoid."""
    policy = make_continuation("policy_consistent")
    away = rollout(world, chunk("x-40mm"), -1.0, HORIZON, world.config,
                   continuation=policy, depth=DEPTH)
    assert away.actions[0] == "x-40mm"
    assert away.actions[1:] != ["x-40mm", "x-40mm"], (
        "the policy repeated a retreating action"
    )
    repeated = rollout(world, chunk("x-40mm"), -1.0, HORIZON, world.config, depth=DEPTH)
    assert repeated.actions == ["x-40mm"] * DEPTH
    # Same first step, different futures: that is the whole point.
    assert away.checkpoints[1] == repeated.checkpoints[1]
    assert away.checkpoints[-1] != repeated.checkpoints[-1]


def test_policy_continuation_is_stateful_across_steps(world):
    """Each continuation choice must come from the state the previous one made."""
    policy = make_continuation("policy_consistent")
    result = rollout(world, chunk("x+40mm", 4), -1.0, 1.6, world.config,
                     continuation=policy, depth=4)
    steps = [r["step"] for r in result.continuation_reasons]
    assert steps == list(range(len(result.actions)))
    # Later choices see different feasible sets than the first branch state.
    feasibles = [r.get("feasible") for r in result.continuation_reasons if "feasible" in r]
    assert feasibles, "policy continuation recorded no branch-state evaluation"


def test_policy_continuation_differs_between_two_branch_states(world):
    """Same policy, two genuinely different states, and it is allowed to differ."""
    from jev_libero.world import Snapshot

    policy = make_continuation("policy_consistent")
    snap = Snapshot(world)
    try:
        first, _ = policy.choose_next_action(world, "x+40mm", -1.0, 1, world.config)
        # Advance the live state, then ask again from somewhere else entirely.
        for _ in range(3):
            world.execute("z-40mm", -1.0)
        second, _ = policy.choose_next_action(world, "x+40mm", -1.0, 1, world.config)
    finally:
        snap.restore()
    assert isinstance(first, str) and isinstance(second, str)
    # Not asserting inequality (physics may agree); asserting the decision is
    # recomputed rather than copied is done by the divergence test above.
    assert first in ("x+40mm", "x+10mm", "x+3mm", "hold")
    assert second in ("x+40mm", "x+10mm", "x+3mm", "hold")


# --- Test 4: determinism ------------------------------------------------------


def test_policy_continuation_is_deterministic(world):
    policy = make_continuation("policy_consistent")
    first = rollout(world, chunk("x+40mm"), -1.0, HORIZON, world.config,
                    continuation=policy, depth=DEPTH)
    second = rollout(world, chunk("x+40mm"), -1.0, HORIZON, world.config,
                     continuation=policy, depth=DEPTH)
    assert first.actions == second.actions
    assert first.checkpoints == second.checkpoints
    assert first.continuation_reasons == second.continuation_reasons


def test_continuation_does_not_contaminate_the_live_environment(world):
    """Isolation still holds with a policy that simulates inside the rollout."""
    policy = make_continuation("policy_consistent")
    before = {
        "qpos": world.data.qpos.copy(),
        "qvel": world.data.qvel.copy(),
        "state": world.env.sim.get_state().flatten().copy(),
        "time": float(world.data.time),
    }
    rollout(world, chunk("x+40mm"), -1.0, HORIZON, world.config,
            continuation=policy, depth=DEPTH)
    np.testing.assert_allclose(before["qpos"], world.data.qpos, atol=0.0, rtol=0)
    np.testing.assert_allclose(before["qvel"], world.data.qvel, atol=0.0, rtol=0)
    np.testing.assert_allclose(
        before["state"], world.env.sim.get_state().flatten(), atol=0.0, rtol=0
    )
    assert float(world.data.time) == before["time"]


# --- Test 7: full and compact come from the same rollout ----------------------


def test_full_and_compact_describe_the_same_physics(world):
    before, predictions = world.predict_all(-1.0)
    feasible, _ = hard_feasible(before, predictions, world.config)
    names = sorted(feasible)[:4]

    def run(representation):
        subset = {name: dict(feasible[name]) for name in names}
        steps, _, diag = attach_trajectories(
            world, subset, -1.0, DEPTH, world.config, HORIZON,
            "policy_consistent", representation,
        )
        return subset, steps, diag

    full, full_steps, full_diag = run("full")
    compact, compact_steps, compact_diag = run("compact")
    assert full_steps == compact_steps
    for name in names:
        assert full_diag[name]["simulated_actions"] == compact_diag[name]["simulated_actions"]
        assert full[name]["trajectory"]["horizon_s"] == compact[name]["trajectory"]["horizon_s"]
        assert "checkpoints" in full[name]["trajectory"]
        assert "paths" in compact[name]["trajectory"]
        assert contains_oracle(compact[name]["trajectory"]) == []
        assert contains_oracle(full[name]["trajectory"]) == []


def test_compact_uses_fewer_tokens_on_real_rollouts(world):
    from jev_libero.experiment import approx_tokens

    before, predictions = world.predict_all(-1.0)
    feasible, _ = hard_feasible(before, predictions, world.config)
    subset = {name: dict(feasible[name]) for name in sorted(feasible)[:6]}
    attach_trajectories(world, subset, -1.0, DEPTH, world.config, HORIZON,
                        "policy_consistent", "full")
    full_tokens = sum(approx_tokens(p["trajectory"]) for p in subset.values())
    subset2 = {name: dict(feasible[name]) for name in sorted(feasible)[:6]}
    attach_trajectories(world, subset2, -1.0, DEPTH, world.config, HORIZON,
                        "policy_consistent", "compact")
    compact_tokens = sum(approx_tokens(p["trajectory"]) for p in subset2.values())
    assert compact_tokens < full_tokens


# --- Test 2 / 6 / 9: the arms, end to end ------------------------------------


@pytest.mark.parametrize(
    "continuation,representation,shuffle",
    [
        ("repeat", "full", False),
        ("hold", "full", False),
        ("policy_consistent", "full", False),
        ("policy_consistent", "compact", False),
        ("policy_consistent", "compact", True),
    ],
)
def test_stage15_arm_runs_end_to_end(simulator, tmp_path, monkeypatch,
                                     continuation, representation, shuffle):
    import jev_libero.runner as runner

    monkeypatch.setattr(runner, "Decisions", lambda *a, **k: MockJev())
    out = tmp_path / f"{continuation}_{representation}_{int(shuffle)}"
    result = runner.run(
        "top_drawer", out, seed=1, render=False, mode="strong_policy_future",
        max_decisions=2, candidate_depth=DEPTH, future_horizon_s=HORIZON,
        chunk_continuation=continuation, future_representation=representation,
        shuffle_future_arm=shuffle,
    )
    assert result["experiment_mode"] == "strong_policy_future"
    assert result["chunk_continuation"] == continuation
    assert result["future_representation"] == representation
    assert result["estimated_tokens_per_decision"] > 0
    assert result["estimated_future_tokens_per_decision"] > 0
    assert set(result["decisions_by_phase"]) == {"far", "mid", "near", "terminal"}
    rows = [json.loads(x) for x in (out / "continuation.jsonl").read_text().splitlines()]
    assert len(rows) == result["decisions"]
    assert all(r["continuation"] == continuation for r in rows)


def test_receding_horizon_still_executes_one_primitive(simulator, tmp_path, monkeypatch):
    """Test 6: policy futures simulate three actions; the live env takes one."""
    import jev_libero.runner as runner

    monkeypatch.setattr(runner, "Decisions", lambda *a, **k: MockJev())
    out = tmp_path / "receding"
    result = runner.run(
        "top_drawer", out, seed=1, render=False, mode="strong_policy_future",
        max_decisions=3, candidate_depth=DEPTH, future_horizon_s=HORIZON,
        chunk_continuation="policy_consistent",
    )
    rows = [json.loads(x) for x in (out / "continuation.jsonl").read_text().splitlines()]
    trace = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines()]
    for row, step in zip(rows, trace):
        simulated = row["candidates"][step["choice"]]["simulated_actions"]
        assert len(simulated) == DEPTH
        assert simulated[0] == step["choice"]
    assert result["sim_steps"] <= result["decisions"] * 8


def test_strong_baseline_is_not_disturbed(simulator, tmp_path):
    """Test 9: S1 still runs without a trajectory and without the verdict.

    Uses the deterministic mock provider rather than an in-memory stub, so the
    request log it writes is the one actually sent.
    """
    import jev_libero.runner as runner

    out = tmp_path / "s1"
    result = runner.run(
        "top_drawer", out, seed=1, render=False, mode="original_no_oracle",
        max_decisions=2, provider="mock",
    )
    assert result["experiment_mode"] == "original_no_oracle"
    assert result["chunk_continuation"] is None
    assert result["future_representation"] is None
    assert not (out / "continuation.jsonl").exists()
    assert result["estimated_future_tokens_per_decision"] == 0
    calls = [json.loads(x) for x in (out / "api.jsonl").read_text().splitlines()]
    for call in calls:
        criteria = call["request"]["questions"][call["layer"]]["criteria"]
        assert contains_oracle(criteria) == []
        assert all("future_trajectory" not in o for o in criteria.values())
