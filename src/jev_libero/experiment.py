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
    # Stage 1.5: the strong one-step baseline plus a future, differing from it
    # in exactly one thing. S1 == original_no_oracle; the arms below add a
    # trajectory on top without touching the one-step fields.
    "strong_policy_future",
)

# Stage 1.5 arms, by the names the report uses.
STAGE15_ARMS = {
    "S1": {"mode": "original_no_oracle"},
    "S1+Repeat": {"mode": "strong_policy_future", "continuation": "repeat",
                  "representation": "full"},
    "S1+Hold": {"mode": "strong_policy_future", "continuation": "hold",
                "representation": "full"},
    "S1+PolicyFull": {"mode": "strong_policy_future", "continuation": "policy_consistent",
                      "representation": "full"},
    "S1+PolicyCompact": {"mode": "strong_policy_future", "continuation": "policy_consistent",
                         "representation": "compact"},
    "S1+PolicyShuffle": {"mode": "strong_policy_future", "continuation": "policy_consistent",
                         "representation": "compact", "shuffle": True},
}
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


def compact_future(result, task_config):
    """Only what the trajectory adds beyond the one-step fields beside it.

    Absolute state repeated at every checkpoint is mostly the same numbers three
    times. This reports each quantity as its path, and each contact fact as its
    transitions, so the added tokens carry added information.
    """
    points = result.checkpoints
    if len(points) < 2:
        return None
    tracked = ["distance_to_moving_geometry_mm", "finger_gap_mm"]
    tracked += [
        key
        for key in (task_config or {}).get("features", {})
        if key in points[0] and isinstance(points[0][key], (int, float))
    ]
    paths = {}
    for key in tracked:
        series = [round(float(point[key]), 2) for point in points if key in point]
        if len(set(series)) > 1:
            paths[key] = series
    contact = [bool(point["moving_contact"]) for point in points]
    obstacle = [bool(point["obstacle_contact"]) for point in points]
    summary = {
        "t_s": [round(point["t_s"], 2) for point in points],
        "paths": paths,
    }
    if len(set(contact)) > 1:
        summary["target_contact"] = contact
    else:
        summary["target_contact_throughout"] = contact[0]
    if any(obstacle):
        summary["non_target_contact"] = obstacle
    if result.events:
        summary["events"] = [f"{e['event']}@{e['t_s']}s" for e in result.events]
    return summary


def future_novelty(result, immediate):
    """Diagnostic only: what the trajectory shows that the first step did not.

    Never enters a prompt and never ranks a candidate. It exists so the analysis
    can say whether a longer future carried anything new at all.
    """
    points = result.checkpoints
    if len(points) < 2:
        return {"future_novelty_count": 0, "signals": []}
    first, last = points[0], points[-1]
    one = points[1] if len(points) > 1 else last
    signals = []
    if bool(one["moving_contact"]) != bool(last["moving_contact"]):
        signals.append("target_contact_changes_after_first_step")
    if bool(one["obstacle_contact"]) != bool(last["obstacle_contact"]):
        signals.append("non_target_contact_changes_after_first_step")
    for key in ("distance_to_moving_geometry_mm",):
        if key in first and key in one and key in last:
            early = one[key] - first[key]
            late = last[key] - one[key]
            if early * late < 0:
                signals.append(f"{key}_reverses")
            elif abs(late) > abs(early) * 1.5:
                signals.append(f"{key}_accelerates")
    if len(result.events) > sum(1 for e in result.events if e["t_s"] <= one["t_s"] + 1e-9):
        signals.append("events_appear_after_first_step")
    if len(set(result.actions)) > 1:
        signals.append("continuation_diverges_from_first_input")
    return {"future_novelty_count": len(signals), "signals": signals}


def attach_trajectories(
    world,
    predictions,
    grip,
    depth,
    task_config,
    horizon=None,
    continuation="repeat",
    representation="full",
):
    """Add a `trajectory` field to each prediction, in place.

    The published pipeline's derived fields keep describing the single primitive
    that actually executes; this only adds the path that follows it. That is the
    difference `original_deep` got wrong — it replaced the executable step with
    a chunk endpoint, and lost every intermediate state.

    Returns (simulated_steps, latency_ms, diagnostics) for the decision record.
    """
    from .continuation import make_continuation
    from .futures import CandidateSequence, chunk_actions, rollout_all

    policy = make_continuation(continuation)
    candidates = [
        CandidateSequence(
            candidate_id=name,
            actions=chunk_actions(name, depth, "repeat"),
            family="",
            first_input=name,
        )
        for name in predictions
    ]
    rollouts = rollout_all(
        world, candidates, grip, horizon, task_config, continuation=policy, depth=depth
    )
    diagnostics = {}
    for name, result in rollouts.items():
        if representation == "compact":
            body = compact_future(result, task_config)
        else:
            # Checkpoint 0 is the current state, already in the prompt's `before`.
            body = {"checkpoints": result.checkpoints[1:], "events": result.events}
        predictions[name]["trajectory"] = {
            "horizon_s": round(result.horizon_seconds, 3),
            "continuation": continuation,
            **(body or {}),
        }
        diagnostics[name] = {
            "simulated_actions": result.actions,
            "continuation_reasons": result.continuation_reasons,
            **future_novelty(result, predictions[name]),
        }
    return (
        sum(r.horizon_steps for r in rollouts.values()),
        round(sum(r.rollout_latency_ms for r in rollouts.values()), 3),
        diagnostics,
    )


def approx_tokens(value):
    """Character-based token estimate. Marked estimated wherever it is reported.

    The provider returns only a total, so the split between base prompt,
    immediate consequence and future trajectory is approximated here rather
    than measured. Ratios between arms are the usable part, not absolutes.
    """
    import json

    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, round(len(text) / 4))


def token_budget(state, instructions, criteria):
    """Estimated split of one request into base / immediate / future."""
    future, immediate = 0, 0
    for option in (criteria or {}).values():
        if not isinstance(option, dict):
            continue
        for key, body in option.items():
            if key in ("future", "future_trajectory", "trajectory"):
                future += approx_tokens(body)
            else:
                immediate += approx_tokens({key: body})
    return {
        "estimated": True,
        "base_tokens": approx_tokens(state) + approx_tokens(instructions),
        "immediate_tokens": immediate,
        "future_tokens": future,
        "total_tokens": approx_tokens(state)
        + approx_tokens(instructions)
        + immediate
        + future,
    }


class TokenAccountingAPI:
    """Records the estimated prompt split for every request, then forwards it."""

    def __init__(self, inner):
        self.inner = inner
        self.budgets = []

    @property
    def total(self):
        return self.inner.total

    @property
    def calls(self):
        return self.inner.calls

    def choose(self, step, layer, state, instructions, criteria):
        self.budgets.append(
            {"step": step, "layer": layer, **token_budget(state, instructions, criteria)}
        )
        return self.inner.choose(step, layer, state, instructions, criteria)

    def close(self):
        self.inner.close()


class ShufflingTrajectoryAPI:
    """Deranges trajectory-to-candidate correspondence just before transport.

    Everything else — the candidate set, the one-step fields, the generated
    futures, the schema and the token volume — is identical to the arm it is
    the control for. Only which future sits beside which action changes.
    """

    def __init__(self, inner, seed=0):
        self.inner = inner
        self.seed = seed
        self.assignments = []

    @property
    def total(self):
        return self.inner.total

    @property
    def calls(self):
        return self.inner.calls

    def choose(self, step, layer, state, instructions, criteria):
        from .futures import derangement

        carriers = [
            name
            for name, option in (criteria or {}).items()
            if isinstance(option, dict) and "future_trajectory" in option
        ]
        if len(carriers) > 1:
            mapping = derangement(carriers, self.seed + step)
            futures = {name: criteria[name]["future_trajectory"] for name in carriers}
            criteria = {
                name: (
                    {**option, "future_trajectory": futures[mapping[name]]}
                    if name in carriers
                    else option
                )
                for name, option in criteria.items()
            }
            self.assignments.append({"step": step, "layer": layer, "mapping": mapping})
        return self.inner.choose(step, layer, state, instructions, criteria)

    def close(self):
        self.inner.close()


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
    continuation="repeat",
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
    candidates = generate_candidates(
        predictions, feasible, candidate_count, candidate_depth, continuation
    )
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
        "chunk_continuation": continuation,
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
