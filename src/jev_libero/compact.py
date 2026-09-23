"""One serializer for the compact prompt, shared by every arm that uses it.

Stage 1.5 left a confound: `policy_compact` compacted only its *future*, while
its immediate consequence stayed in the verbose form `s1_original` uses. So
`S1Original → PolicyCompact` changed two things at once — representation and
future information — and the 26-vs-32 result could not be attributed.

Stage 2 splits them. `s1_compact` carries exactly the facts `s1_original`
carries, formatted by `compact_immediate`; `policy_compact` carries the same
thing plus a compact future. The step from one to the other is then the future
and nothing else.

The compaction is lossless by construction: every field keeps its value, and the
field names move to a legend sent once in the state instead of being repeated on
all 27 candidates. `LEGEND` is the mapping, and the equivalence test reads it
back to prove no fact was dropped or added.
"""

# Terse keys, and what each one means. Sent once per request in the state.
LEGEND = {
    "gap_red_mm": "predicted_surface_gap_reduction_mm",
    "closing_mm": "predicted_closing_mm",
    "tgt_contact": "predicted_target_contact",
    "obst_force_N": "predicted_obstacle_peak_force_N",
    "lift_mm": "predicted_lift_mm",
    "goal_dist_mm": "predicted_goal_distance_mm",
    "angle_deg": "predicted_opening_degrees",
}
INVERSE_LEGEND = {long: short for short, long in LEGEND.items()}

# Kept verbatim: it is the action's meaning, not a measurement.
PASSTHROUGH = ("input",)


def shorten(field):
    """Terse key for a task's derived field, or the field itself if unmapped."""
    return INVERSE_LEGEND.get(field, field)


def compact_immediate(option):
    """Compact form of one candidate's one-step consequence.

    Takes the option dict the published pipeline would have sent and returns the
    same facts under legend keys. No field is dropped, no value is changed, and
    nothing about the future is added.
    """
    compact = {}
    for field, value in option.items():
        if field in PASSTHROUGH:
            compact[field] = value
        elif field in ("future_trajectory", "trajectory"):
            continue  # handled by the caller; never folded into the immediate part
        else:
            compact[shorten(field)] = value
    return compact


def expand_immediate(compact_option):
    """Recover the long-form facts from a compact option, for the equivalence test."""
    expanded = {}
    for field, value in compact_option.items():
        if field in PASSTHROUGH:
            expanded[field] = value
        elif field in ("future_trajectory", "trajectory"):
            continue
        else:
            expanded[LEGEND.get(field, field)] = value
    return expanded


def legend_for(options):
    """The subset of the legend the given options actually use.

    Sent in the state so the terse keys are self-describing without repeating
    the long names on every candidate.
    """
    used = set()
    for option in options.values():
        if isinstance(option, dict):
            used.update(option)
    return {short: long for short, long in LEGEND.items() if short in used}


def compact_options(options, keep_future=True):
    """Compact every candidate. `keep_future` is False for the S1 arms.

    The future is attached beside the immediate part rather than merged into it,
    so an arm that must not see a future simply never gets the key.
    """
    result = {}
    for name, option in options.items():
        if not isinstance(option, dict):
            result[name] = option
            continue
        entry = compact_immediate(option)
        future = option.get("future_trajectory") or option.get("trajectory")
        if keep_future and future is not None:
            entry["future"] = future
        result[name] = entry
    return result


class CompactFormattingAPI:
    """Rewrites requests into the compact form just before transport.

    Formatting happens at the boundary so `ValidatedPolicy` is untouched and the
    arms differ only in which wrapper they are given. `keep_future=False` also
    makes the S1 arms structurally incapable of carrying a trajectory: the key is
    dropped even if something upstream attached one.
    """

    def __init__(self, inner, keep_future=True):
        self.inner = inner
        self.keep_future = keep_future

    @property
    def total(self):
        return self.inner.total

    @property
    def calls(self):
        return self.inner.calls

    def choose(self, step, layer, state, instructions, criteria):
        compact = compact_options(criteria or {}, keep_future=self.keep_future)
        legend = legend_for(compact)
        if legend:
            state = {**state, "field_names": legend}
        return self.inner.choose(step, layer, state, instructions, compact)

    def close(self):
        self.inner.close()
