"""Counterfactual future rollouts over the existing reversible MuJoCo branches.

MuJoCo is used as an oracle / simulator-based world model. It is NOT a learned
neural world model. Rollouts reuse `World.execute` and `Snapshot`, so branches
run exactly the same 27 atomic inputs as live execution.

Future summaries carry objective physical consequences only. The LIBERO success
predicate, task rewards, and any ranking verdict are withheld: deciding whether
a future is good is Jev's job, not the simulator's.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from .actions import ACTIONS
from .policy import blocked_pairs, family

# Substrings forbidden anywhere in a future summary handed to Jev. Keeping the
# simulator's own verdict out of the prompt is what makes the study meaningful.
ORACLE_MARKERS = (
    "success",
    "reward",
    "task_score",
    "oracle",
    "best_action",
    "good_action",
    "best_candidate",
    "recommended",
    "can_finish",
    "complete",
    "progress",
    "verdict",
)

# Objective state exposed at every checkpoint. Task feature values are added on
# top of these, minus anything matching ORACLE_MARKERS.
CHECKPOINT_FIELDS = (
    "eef_mm",
    "finger_gap_mm",
    "moving_contact",
    "obstacle_contact",
    "contact_modes",
    "target_force_N",
    "obstacle_force_N",
    "distance_to_moving_geometry_mm",
)


def contains_oracle(node, path=""):
    """Return the paths of any oracle-flavoured key or string inside `node`."""
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if any(marker in str(key).lower() for marker in ORACLE_MARKERS):
                hits.append(here)
            hits.extend(contains_oracle(value, here))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            hits.extend(contains_oracle(value, f"{path}[{index}]"))
    elif isinstance(node, str):
        if any(marker in node.lower() for marker in ORACLE_MARKERS):
            hits.append(path)
    return hits


def strip_oracle(features, task_config=None):
    """Objective subset of a feature dictionary, with the oracle verdict removed."""
    clean = {}
    for key in CHECKPOINT_FIELDS:
        if key in features:
            clean[key] = features[key]
    for key, value in (task_config or {}).get("features", {}).items():
        if any(marker in key.lower() for marker in ORACLE_MARKERS):
            continue
        if key in features and isinstance(features[key], (int, float, bool, list)):
            clean[key] = features[key]
    return clean


@dataclass
class CandidateSequence:
    """One short action chunk. Only `actions[0]` is ever executed for real."""

    candidate_id: str
    actions: list
    family: str
    first_input: str

    def as_record(self):
        return {
            "candidate_id": self.candidate_id,
            "actions": list(self.actions),
            "family": self.family,
            "first_input": self.first_input,
        }


@dataclass
class CounterfactualRollout:
    """Trajectory-level future of one candidate, measured in the live simulator."""

    candidate_id: str
    actions: list
    horizon_steps: int = 0
    horizon_seconds: float = 0.0
    primitives_simulated: int = 0
    checkpoints: list = field(default_factory=list)
    events: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    rollout_latency_ms: float = 0.0

    def as_record(self):
        return {
            "candidate_id": self.candidate_id,
            "actions": list(self.actions),
            "simulated_steps": self.horizon_steps,
            "duration_s": round(self.horizon_seconds, 6),
            "primitives_simulated": self.primitives_simulated,
            "latency_ms": round(self.rollout_latency_ms, 3),
            "checkpoints": self.checkpoints,
            "events": self.events,
            "summary": self.summary,
        }


def allocate(sizes, count):
    """Largest-remainder allocation of `count` slots proportional to `sizes`.

    Round-robin over families would over-represent tiny families: with 18
    translations, 6 rotations, 2 finger inputs and 1 hold, it handed Jev a menu
    that was five-sixths non-translation and excluded every large approach move.
    Proportional allocation keeps the menu shaped like the feasible pool.
    """
    total = sum(sizes.values())
    exact = {key: size * count / total for key, size in sizes.items()}
    floors = {key: min(int(value), sizes[key]) for key, value in exact.items()}
    # Every family keeps at least one slot while slots remain, so the menu still
    # spans the available motion kinds.
    for key in sorted(sizes, key=lambda k: (-sizes[k], k)):
        if sum(floors.values()) >= count:
            break
        if floors[key] == 0:
            floors[key] = 1
    remaining = count - sum(floors.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - floors[k]), -sizes[k], k))
    index = 0
    while remaining > 0 and any(floors[k] < sizes[k] for k in sizes):
        key = order[index % len(order)]
        if floors[key] < sizes[key]:
            floors[key] += 1
            remaining -= 1
        index += 1
    while sum(floors.values()) > count:
        key = max(sorted(floors), key=lambda k: (floors[k], -sizes[k]))
        floors[key] -= 1
    return floors


def spread(members, slots):
    """Evenly spaced picks across an ordered family, endpoints included."""
    if slots >= len(members):
        return list(members)
    if slots == 1:
        return [members[0]]
    step = (len(members) - 1) / (slots - 1)
    return [members[round(i * step)] for i in range(slots)]


def generate_candidates(predictions, feasible, count, depth):
    """Deterministic, mode-independent candidate chunks.

    Selection never reads the LIBERO reward, the success predicate, or any task
    score — only the feasible set, the predicted motion family, and the fixed
    `ACTIONS` ordering. Slots are allocated proportionally across families and
    spread within each family, so the menu covers the action space rather than
    the first few entries of it. The result is identical for every experiment
    mode given the same state.
    """
    if count < 1 or depth < 1:
        raise ValueError("Use a positive candidate count and depth.")
    order = [name for name in ACTIONS if name in feasible]
    grouped = {}
    for name in order:
        grouped.setdefault(family(name, predictions[name]), []).append(name)
    sizes = {key: len(value) for key, value in grouped.items()}
    slots = allocate(sizes, min(count, len(order)))
    picked = [name for key in sorted(grouped) for name in spread(grouped[key], slots[key])]
    picked.sort(key=order.index)
    return [
        CandidateSequence(
            candidate_id=f"C{index + 1}",
            actions=[name] * depth,
            family=family(name, predictions[name]),
            first_input=name,
        )
        for index, name in enumerate(picked)
    ]


def hard_feasible(before, predictions, task_config):
    """Physical admissibility only; identical in every experiment mode.

    These are the original project's collision rules, not a future-aware ranking.
    """
    rules = task_config["contact"]
    blocked = blocked_pairs(before)
    feasible, rejected = {}, {}
    for name, p in predictions.items():
        if rules["reject_new_obstacles"] and set(map(tuple, p["obstacle_pairs_seen"])) - blocked:
            rejected[name] = "introduces non-target contact at sampled control steps"
            continue
        if rules["require_obstacle_free_endpoint"] and p["after"]["obstacle_contact"]:
            rejected[name] = "does not resolve existing non-target contact"
            continue
        feasible[name] = p
    return feasible, rejected


def rollout(world, candidate, grip, horizon_seconds, task_config, snapshot=None):
    """Simulate one candidate chunk from `snapshot`, then restore the live state.

    The live environment is always restored, including on error, so no
    counterfactual can contaminate the real episode.
    """
    import time

    from .world import Snapshot

    root = snapshot or Snapshot(world)
    start_time = float(world.data.time)
    began = time.perf_counter()
    initial = world.features()
    initial_blocked = blocked_pairs(initial)
    checkpoints = [
        {
            "t_s": 0.0,
            "after_primitive": 0,
            "input": None,
            **strip_oracle(initial, task_config),
        }
    ]
    events = []
    steps = 0
    primitives = 0
    # Measured before the restore below: restoring rewinds world.data.time too.
    elapsed = 0.0
    try:
        current_grip = grip
        previous = initial
        for index, name in enumerate(candidate.actions):
            if horizon_seconds is not None and float(world.data.time) - start_time >= (
                horizon_seconds - 1e-9
            ):
                break
            result = world.execute(name, current_grip)
            current_grip = result["grip"]
            steps += result["steps"]
            primitives += 1
            elapsed = float(world.data.time) - start_time
            features = result["features"]
            checkpoints.append(
                {
                    "t_s": round(elapsed, 4),
                    "after_primitive": index + 1,
                    "input": name,
                    **strip_oracle(features, task_config),
                }
            )
            new_pairs = set(result["obstacle_pairs_seen"]) - initial_blocked
            if new_pairs:
                events.append(
                    {
                        "t_s": round(elapsed, 4),
                        "event": "non_target_contact",
                        "pairs": sorted(tuple(pair) for pair in new_pairs),
                    }
                )
            if previous["moving_contact"] and not features["moving_contact"]:
                events.append(
                    {"t_s": round(elapsed, 4), "event": "target_contact_lost"}
                )
            if not previous["moving_contact"] and features["moving_contact"]:
                events.append(
                    {"t_s": round(elapsed, 4), "event": "target_contact_established"}
                )
            previous = features
    finally:
        root.restore()
    final = checkpoints[-1]
    first = checkpoints[0]
    summary = {
        "net_eef_displacement_mm": round(
            float(np.linalg.norm(np.array(final["eef_mm"]) - np.array(first["eef_mm"]))), 3
        ),
        "ends_in_target_contact": bool(final["moving_contact"]),
        "ends_in_non_target_contact": bool(final["obstacle_contact"]),
        "peak_non_target_force_N": round(
            max(point["obstacle_force_N"] for point in checkpoints), 3
        ),
        "event_count": len(events),
    }
    if "distance_to_moving_geometry_mm" in first:
        summary["net_surface_gap_change_mm"] = round(
            final["distance_to_moving_geometry_mm"] - first["distance_to_moving_geometry_mm"], 3
        )
    for key in (task_config or {}).get("features", {}):
        if key in first and key in final and isinstance(first[key], (int, float)):
            summary[f"net_{key}_change"] = round(float(final[key]) - float(first[key]), 4)
    return CounterfactualRollout(
        candidate_id=candidate.candidate_id,
        actions=list(candidate.actions),
        horizon_steps=steps,
        horizon_seconds=round(elapsed, 6),
        primitives_simulated=primitives,
        checkpoints=checkpoints,
        events=events,
        summary=summary,
        rollout_latency_ms=(time.perf_counter() - began) * 1000,
    )


def rollout_all(world, candidates, grip, horizon_seconds, task_config):
    """Roll out every candidate from one shared snapshot of the live state."""
    from .world import Snapshot

    root = Snapshot(world)
    rollouts = {}
    try:
        for candidate in candidates:
            root.restore()
            rollouts[candidate.candidate_id] = rollout(
                world, candidate, grip, horizon_seconds, task_config, snapshot=root
            )
    finally:
        root.restore()
    return rollouts


def derangement(ids, seed):
    """Deterministic permutation with no fixed point; the negative control."""
    if len(ids) < 2:
        raise ValueError("A derangement needs at least two candidates.")
    rng = random.Random(seed)
    shuffled = list(ids)
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(ids, shuffled)):
            return dict(zip(ids, shuffled))
    raise RuntimeError("Could not build a derangement for the candidate set.")
