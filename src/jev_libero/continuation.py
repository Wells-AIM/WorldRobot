"""How a counterfactual future continues past the action that will actually run.

Under receding-horizon control the live environment only ever receives `a_t`;
everything after it in a simulated chunk is an assumption about what the robot
would do next. That assumption is not neutral.

Stage-1 measured the cost of getting it wrong. `repeat` shows `a_t` performed D
times, so a 40 mm input appears as 120 mm of travel and saturates: over an
episode the 3-step approach was 2.67x the 1-step figure for 3 mm inputs but only
1.04x for 40 mm ones, compressing a real 2.66x advantage of a large step over a
small one down to 1.03x. `hold` avoids that but its endpoint then duplicates the
one-step field it sits beside, adding prompt volume and no information.

`policy_consistent` asks the question the control loop actually asks: execute
`a_t`, look at the state that produces, and choose the next action from there.

No policy here trains anything, calls a model, or reads the LIBERO reward, the
success predicate, or any oracle score.
"""

from .actions import ACTIONS
from .config import expression, project

# Continuations that need no branch-state evaluation.
STATIC_CONTINUATIONS = ("repeat", "hold")
CONTINUATIONS = (*STATIC_CONTINUATIONS, "policy_consistent")


class OracleAccess(RuntimeError):
    """A continuation tried to read the simulator's verdict."""


def neutralize_oracle(features):
    """A feature dict with the success predicate forced false.

    The task contracts are reused to score continuations, but `advance_target`
    in the bundled tasks reads `{"any": [closing_mm >= threshold, after.success]}`.
    Forcing the predicate false keeps the physical branch and drops the oracle
    branch, which is the split the contracts themselves do not make.
    """
    if "success" not in features:
        return features
    return {**features, "success": False}


def continuation_actions(first_input):
    """The actions a branch state is allowed to continue with.

    Deliberately narrow and derived only from the first input's own shape: keep
    going at this scale, back off to a finer scale on the same axis, or stop.
    That is the choice `repeat` cannot make, and it is what a large step needs
    once it has begun to overshoot. Nothing task-specific, nothing learned.
    """
    spec = ACTIONS[first_input]
    if "axis" in spec:
        axis = "xyz"[spec["axis"]]
        sign = "+" if spec["mm"] > 0 else "-"
        magnitudes = [40, 10, 3]
        same = [f"{axis}{sign}{mm}mm" for mm in magnitudes]
        # Only scales at or below the first input's own; never escalate.
        keep = [name for name in same if abs(ACTIONS[name]["mm"]) <= abs(spec["mm"])]
        return [*keep, "hold"]
    return [first_input, "hold"]


class ContinuationPolicy:
    """Chooses the action a simulated branch takes after its first input."""

    name = "abstract"
    reads_branch_state = False

    def choose_next_action(self, world, first_input, grip, step_index, task_config):
        raise NotImplementedError

    def as_record(self):
        return {"continuation": self.name, "reads_branch_state": self.reads_branch_state}


class RepeatContinuation(ContinuationPolicy):
    """Do the same thing again. Cheap, and wrong for large steps."""

    name = "repeat"

    def choose_next_action(self, world, first_input, grip, step_index, task_config):
        return first_input, {"rule": "repeat_first_input"}


class HoldContinuation(ContinuationPolicy):
    """Let the action settle. Avoids saturation; adds little."""

    name = "hold"

    def choose_next_action(self, world, first_input, grip, step_index, task_config):
        return "hold", {"rule": "hold_after_first_input"}


class PolicyConsistentContinuation(ContinuationPolicy):
    """Re-decide from the branch state, the way the control loop would.

    At each branch state it previews a narrow set of continuations, applies the
    task's own collision rules, evaluates the task's effect contracts with the
    success predicate neutralized, and takes the best by the task's own score
    with its oracle term removed. Ties break on the fixed `ACTIONS` order, so
    the whole thing is deterministic given a snapshot, a candidate and a horizon.
    """

    name = "policy_consistent"
    reads_branch_state = True

    def choose_next_action(self, world, first_input, grip, step_index, task_config):
        from .futures import hard_feasible
        from .policy import contracts
        from .world import Snapshot

        options = continuation_actions(first_input)
        branch = Snapshot(world)
        before = world.features()
        previews = {}
        try:
            for name in options:
                branch.restore()
                result = world.execute(name, grip)
                after = result["features"]
                previews[name] = {
                    "after": after,
                    **project(task_config["predictions"], {"before": before, "after": after}),
                    "geometry_approach_mm": before["distance_to_moving_geometry_mm"]
                    - after["distance_to_moving_geometry_mm"],
                    "eef_displacement_mm": 0.0,
                    "gap_change_mm": after["finger_gap_mm"] - before["finger_gap_mm"],
                    "peak_obstacle_force_N": result["peak_obstacle_force_N"],
                    "obstacle_pairs_seen": result["obstacle_pairs_seen"],
                    "steps": result["steps"],
                }
        finally:
            branch.restore()

        feasible, rejected = hard_feasible(before, previews, task_config)
        if not feasible:
            return "hold", {
                "rule": "no_feasible_continuation",
                "rejected": len(rejected),
                "considered": options,
            }

        # Effect contracts, with the oracle branch of advance_target removed.
        neutral = {
            name: {**p, "after": neutralize_oracle(p["after"])} for name, p in feasible.items()
        }
        pools, _ = contracts(before, neutral, collision_aware=True, task_config=task_config)
        order = list(task_config["contracts"])
        chosen_effect = next((key for key in order if pools.get(key)), None)
        pool = pools[chosen_effect] if chosen_effect else neutral

        # The task's own ranking, minus its success term.
        rules = [
            rule
            for rule in task_config["search"]["score"]
            if rule != {"ref": "after.success"}
        ]
        best, best_score = None, None
        for name in sorted(pool, key=lambda n: list(ACTIONS).index(n)):
            p = pool[name]
            context = {
                "after": p["after"],
                "before": before,
                "p": p,
                "effects": {k: name in pools.get(k, {}) for k in task_config["contracts"]},
            }
            score = tuple(expression(rule, context) for rule in rules)
            if best_score is None or score > best_score:
                best, best_score = name, score
        return best, {
            "rule": "contract_then_task_score",
            "effect": chosen_effect,
            "considered": options,
            "feasible": sorted(feasible),
            "score_fields": len(rules),
            "step_index": step_index,
        }


POLICIES = {
    "repeat": RepeatContinuation,
    "hold": HoldContinuation,
    "policy_consistent": PolicyConsistentContinuation,
}


def make_continuation(name):
    if name not in POLICIES:
        raise ValueError(f"Unknown continuation policy: {name}; use one of {tuple(POLICIES)}")
    return POLICIES[name]()
