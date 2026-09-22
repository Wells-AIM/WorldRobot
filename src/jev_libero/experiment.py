"""Controlled experiment modes over one shared candidate generator.

Four modes answer a single question: how far into the future should an embodied
agent reason before acting? They share the task, the initial state, the
candidate generator, the candidate count, the action primitives, the hard
feasibility rules, the Jev version, the provider, and the decision budget. The
only core difference is how much future consequence information Jev receives.

    reactive              no future at all
    short_preview         one primitive of consequence (the original 0.4 s depth)
    counterfactual_future trajectory-level consequence to a configurable horizon
    shuffle_future        the same futures, deranged against their candidates

The project's original three-layer pipeline is untouched and still reachable as
mode "original"; see runner.run.
"""

from .actions import ACTIONS
from .futures import (
    ORACLE_MARKERS,
    contains_oracle,
    derangement,
    generate_candidates,
    hard_feasible,
    rollout_all,
    strip_oracle,
)

CONTROLLED_MODES = ("reactive", "short_preview", "counterfactual_future", "shuffle_future")
# Variants of the published pipeline, used to test it against itself rather than
# against a stripped-down baseline. Both keep its three layers, its effect
# contracts and its task-aligned derived fields.
ORIGINAL_MODES = (
    "original",
    "original_no_oracle",
    "original_deep",
    "original_trajectory",
)
MODES = ORIGINAL_MODES + CONTROLLED_MODES

# One primitive of the original engine: 8 environment steps at 20 Hz control.
PRIMITIVE_SECONDS = 0.4

INSTRUCTIONS = {
    "reactive": (
        "Choose one atomic input for the robot from the listed candidates, using the task "
        "and the current physical state only. No consequence information is provided. "
        "Only the chosen input runs; the next decision is recomputed after observing the result."
    ),
    "short_preview": (
        "Each candidate lists the simulated physical consequence of its first primitive. "
        "These are simulator measurements, not judgements: no candidate is marked good or bad. "
        "Decide which consequence best serves the task. Only the chosen input runs; the next "
        "decision is recomputed after observing the result."
    ),
    "counterfactual_future": (
        "Each candidate lists the simulated physical trajectory that follows if the robot keeps "
        "applying that input over the stated horizon, as timed checkpoints plus physical events. "
        "These are simulator measurements, not judgements: no candidate is marked good or bad, and "
        "no task score is supplied. Reason about which future consequence best serves the task, "
        "including consequences that only appear later in the trajectory. Only the first input of "
        "the chosen candidate runs; the next decision is recomputed after observing the result."
    ),
}
INSTRUCTIONS["shuffle_future"] = INSTRUCTIONS["counterfactual_future"]


def prune_oracle(node):
    """Recursively drop keys whose name carries the simulator's own verdict."""
    if isinstance(node, dict):
        return {
            key: prune_oracle(value)
            for key, value in node.items()
            if not any(marker in str(key).lower() for marker in ORACLE_MARKERS)
        }
    if isinstance(node, list):
        return [prune_oracle(value) for value in node]
    return node


class OracleFilteringAPI:
    """Wraps a provider and removes oracle fields from every request.

    The published pipeline shows Jev `predicted_task_complete` and
    `can_finish_task`, both of which are the LIBERO success predicate. Filtering
    at the transport boundary tests how much of that pipeline's strength comes
    from the verdict rather than from its structure, without touching
    `ValidatedPolicy` or the prompts the original mode sends.

    The task effect contracts still consult the predicate when they group
    candidates; only what reaches Jev is filtered.
    """

    def __init__(self, inner):
        self.inner = inner

    @property
    def total(self):
        return self.inner.total

    @property
    def calls(self):
        return self.inner.calls

    def choose(self, step, layer, state, instructions, criteria):
        return self.inner.choose(
            step, layer, prune_oracle(state), instructions, prune_oracle(criteria)
        )

    def close(self):
        self.inner.close()


def attach_trajectories(world, predictions, grip, depth, task_config, horizon=None):
    """Add a `trajectory` field to each prediction, in place.

    The published pipeline's derived fields keep describing the single primitive
    that actually executes; this only adds the path that follows it. That is the
    difference `original_deep` got wrong — it replaced the executable step with
    a chunk endpoint, and lost every intermediate state.

    Returns (simulated_steps, latency_ms) for the decision record.
    """
    from .futures import CandidateSequence, rollout_all

    candidates = [
        CandidateSequence(
            candidate_id=name, actions=[name] * depth, family="", first_input=name
        )
        for name in predictions
    ]
    rollouts = rollout_all(world, candidates, grip, horizon, task_config)
    for name, result in rollouts.items():
        # Checkpoint 0 is the current state, already in the prompt's `before`.
        predictions[name]["trajectory"] = {
            "horizon_s": round(result.horizon_seconds, 3),
            "checkpoints": result.checkpoints[1:],
            "events": result.events,
        }
    return (
        sum(r.horizon_steps for r in rollouts.values()),
        round(sum(r.rollout_latency_ms for r in rollouts.values()), 3),
    )


def horizon_for(mode, requested):
    """Simulated horizon in seconds for a mode; None means 'no simulation'."""
    if mode == "reactive":
        return 0.0
    if mode == "short_preview":
        return PRIMITIVE_SECONDS
    return requested


def decide(
    api,
    world,
    step,
    mode,
    before,
    predictions,
    grip,
    task_config,
    candidate_count=27,
    candidate_depth=3,
    future_horizon_s=1.2,
    shuffle_seed=0,
    history=None,
):
    """One controlled decision. Returns (first_input_to_execute, record)."""
    if mode not in CONTROLLED_MODES:
        raise ValueError(f"Not a controlled experiment mode: {mode}")
    feasible, rejected = hard_feasible(before, predictions, task_config)
    if not feasible:
        from .policy import NoFeasibleAction

        raise NoFeasibleAction(
            "No candidate passes the hard physical feasibility rules; stopping without a fallback."
        )
    candidates = generate_candidates(predictions, feasible, candidate_count, candidate_depth)
    horizon = horizon_for(mode, future_horizon_s)
    rollouts = {}
    if horizon and horizon > 0:
        rollouts = rollout_all(world, candidates, grip, horizon, task_config)
    assignment = {c.candidate_id: c.candidate_id for c in candidates}
    if mode == "shuffle_future" and len(candidates) > 1:
        assignment = derangement([c.candidate_id for c in candidates], shuffle_seed + step)
    options = {}
    for candidate in candidates:
        entry = {
            "input": ACTIONS[candidate.first_input]["description"],
            "motion_family": candidate.family,
        }
        if mode != "reactive":
            entry["planned_chunk"] = list(candidate.actions)
            source = rollouts[assignment[candidate.candidate_id]]
            entry["future"] = {
                "horizon_s": round(source.horizon_seconds, 3),
                "simulated_steps": source.horizon_steps,
                "checkpoints": source.checkpoints,
                "events": source.events,
                **source.summary,
            }
        options[candidate.candidate_id] = entry
    # The task's own feedback projection may carry the success predicate
    # (top_drawer declares one); the controlled arms must not see it.
    outcomes = [
        {key: value for key, value in entry.items() if not contains_oracle({key: value})}
        for entry in (history or [])[-3:]
    ]
    state = {
        "task": task_config["policy"]["task_text"],
        "current_state": strip_oracle(before, task_config),
        "recent_real_outcomes": outcomes,
        "future_information": {
            "reactive": "none",
            "short_preview": "one primitive of simulated consequence per candidate",
            "counterfactual_future": "trajectory-level simulated consequence per candidate",
            "shuffle_future": "trajectory-level simulated consequence per candidate",
        }[mode],
    }
    choice = api.choose(step, "candidate", state, INSTRUCTIONS[mode], options)
    selected = next(c for c in candidates if c.candidate_id == choice)
    record = {
        "experiment_mode": mode,
        "future_horizon_requested_s": horizon,
        "future_horizon_actual_s": round(
            max((r.horizon_seconds for r in rollouts.values()), default=0.0), 4
        ),
        "candidate_depth": candidate_depth,
        "candidate_count": len(candidates),
        "shuffle_seed": shuffle_seed if mode == "shuffle_future" else None,
        "candidates": [
            {
                **candidate.as_record(),
                "original_future_id": candidate.candidate_id,
                "assigned_future_id": assignment[candidate.candidate_id],
                "counterfactual": rollouts[assignment[candidate.candidate_id]].as_record()
                if rollouts
                else None,
            }
            for candidate in candidates
        ],
        "hard_rejected": rejected,
        "jev_request": {"state": state, "instructions": INSTRUCTIONS[mode], "options": options},
        "selected_candidate": choice,
        "executed_input": selected.first_input,
        "rollout_latency_ms": round(sum(r.rollout_latency_ms for r in rollouts.values()), 3),
        "rollout_steps_total": sum(r.horizon_steps for r in rollouts.values()),
    }
    return selected.first_input, record
