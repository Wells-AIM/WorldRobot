"""Stage 2: representation must be separable from information.

`s1_compact` has to carry exactly what `s1_original` carries — no field lost, no
future gained — or the S1Original → S1Compact → PolicyCompact chain stops
isolating one variable per step.
"""

import pytest
from test_futures import MockJev

from jev_libero.compact import (
    LEGEND,
    CompactFormattingAPI,
    compact_immediate,
    compact_options,
    expand_immediate,
    legend_for,
    shorten,
)
from jev_libero.experiment import STAGE2_ARMS, STAGE2_MAIN_ARMS, approx_tokens
from jev_libero.futures import contains_oracle
from jev_libero.records import read_jsonl


def option(**extra):
    base = {
        "input": "Move gripper +3 millimeters along WORLD x; hold orientation and finger command.",
        "predicted_closing_mm": 0.0,
        "predicted_surface_gap_reduction_mm": 2.42,
        "predicted_target_contact": False,
        "predicted_obstacle_peak_force_N": 0.0,
    }
    base.update(extra)
    return base


# --- Section 24: information equivalence --------------------------------------


def test_compact_is_a_lossless_renaming():
    original = option()
    compact = compact_immediate(original)
    assert compact != original, "compaction should change the form"
    assert expand_immediate(compact) == original, "compaction lost or altered a fact"


def test_every_value_survives_unchanged():
    original = option(predicted_closing_mm=1.2345, predicted_target_contact=True)
    compact = compact_immediate(original)
    assert set(compact.values()) >= {1.2345, True, 2.42}
    assert expand_immediate(compact)["predicted_closing_mm"] == 1.2345


def test_compact_adds_no_future_field():
    compact = compact_immediate(option())
    assert not any("future" in key or "trajectory" in key for key in compact)


def test_legend_covers_every_terse_key_used():
    compact = compact_immediate(option())
    legend = legend_for({"x+3mm": compact})
    terse = [k for k in compact if k != "input"]
    assert all(k in legend for k in terse), (terse, legend)
    assert all(legend[k] == LEGEND[k] for k in legend)


def test_unknown_fields_pass_through_rather_than_vanish():
    """A task with a field the legend does not know must not silently lose it."""
    original = option(predicted_something_new_mm=5.0)
    compact = compact_immediate(original)
    assert shorten("predicted_something_new_mm") == "predicted_something_new_mm"
    assert expand_immediate(compact) == original


def test_information_equivalence_on_recorded_requests(records_root):
    """Replay real published prompts through the compact serializer."""
    calls = read_jsonl(records_root / "top_drawer_seed1" / "api.jsonl")
    motor = [c for c in calls if c["layer"] == "motor"]
    assert motor
    for call in motor:
        criteria = call["request"]["questions"]["motor"]["criteria"]
        compact = compact_options(criteria, keep_future=False)
        assert set(compact) == set(criteria), "a candidate disappeared"
        for name in criteria:
            assert expand_immediate(compact[name]) == criteria[name], name


def test_compact_is_smaller_on_recorded_requests(records_root):
    calls = read_jsonl(records_root / "top_drawer_seed1" / "api.jsonl")
    motor = [c for c in calls if c["layer"] == "motor"]
    long_total, short_total = 0, 0
    for call in motor:
        criteria = call["request"]["questions"]["motor"]["criteria"]
        long_total += approx_tokens(criteria)
        short_total += approx_tokens(compact_options(criteria, keep_future=False))
    assert short_total < long_total
    # The legend is sent once per request, not per candidate.
    assert short_total < long_total * 0.95


# --- Section 6: s1_compact must not see a future ------------------------------


class PoisonedFuture(dict):
    """Any read of this trajectory fails the test."""

    def __getitem__(self, key):
        raise AssertionError("s1_compact read a multi-step future")

    def __iter__(self):
        raise AssertionError("s1_compact iterated a multi-step future")


def test_s1_compact_drops_a_future_even_if_one_is_attached():
    """Structural, not conventional: the key cannot reach the provider."""
    criteria = {
        "x+3mm": option(future_trajectory={"paths": {"d": [93.5, 64.4]}}),
        "x+10mm": option(trajectory={"checkpoints": [{"t_s": 0.4}]}),
    }
    compact = compact_options(criteria, keep_future=False)
    for entry in compact.values():
        assert "future" not in entry
        assert "future_trajectory" not in entry
        assert "trajectory" not in entry


def test_s1_compact_never_touches_a_poisoned_future():
    criteria = {"x+3mm": option(future_trajectory=PoisonedFuture())}
    compact = compact_options(criteria, keep_future=False)
    assert "future" not in compact["x+3mm"]
    assert compact["x+3mm"]["gap_red_mm"] == 2.42


def test_policy_compact_does_carry_the_future():
    criteria = {"x+3mm": option(future_trajectory={"paths": {"d": [93.5, 64.4]}})}
    compact = compact_options(criteria, keep_future=True)
    assert compact["x+3mm"]["future"] == {"paths": {"d": [93.5, 64.4]}}
    # and the immediate part is formatted identically to the S1 arm's
    s1 = compact_options({"x+3mm": option()}, keep_future=False)["x+3mm"]
    assert {k: v for k, v in compact["x+3mm"].items() if k != "future"} == s1


# --- Section 7: one serializer, both arms -------------------------------------


def test_the_two_arms_share_the_immediate_serializer():
    """S1Compact -> PolicyCompact must add the future and change nothing else."""
    plain = option()
    withfuture = option(future_trajectory={"paths": {"d": [1.0, 2.0]}})
    s1 = CompactFormattingAPI(MockJev(), keep_future=False)
    policy = CompactFormattingAPI(MockJev(), keep_future=True)
    s1.choose(0, "motor", {}, "pick", {"x+3mm": plain})
    policy.choose(0, "motor", {}, "pick", {"x+3mm": withfuture})
    a = s1.inner.requests[0]["criteria"]["x+3mm"]
    b = policy.inner.requests[0]["criteria"]["x+3mm"]
    assert set(b) - set(a) == {"future"}
    assert {k: v for k, v in b.items() if k != "future"} == a


def test_the_legend_travels_in_the_state_not_on_every_candidate():
    inner = MockJev()
    api = CompactFormattingAPI(inner, keep_future=False)
    api.choose(0, "motor", {"task": "close it"}, "pick", {f"c{i}": option() for i in range(27)})
    request = inner.requests[0]
    assert "field_names" in request["state"]
    assert request["state"]["task"] == "close it"
    assert all("field_names" not in o for o in request["criteria"].values())


def test_compact_formatting_carries_no_oracle():
    inner = MockJev()
    api = CompactFormattingAPI(inner, keep_future=False)
    api.choose(0, "motor", {"task": "x"}, "pick", {"x+3mm": option()})
    assert contains_oracle(inner.requests[0]["criteria"]) == []
    assert contains_oracle(inner.requests[0]["state"]) == []


# --- Arm definitions ----------------------------------------------------------


def test_stage2_arms_differ_one_variable_at_a_time():
    assert STAGE2_MAIN_ARMS == ("s1_original", "s1_compact", "policy_compact",
                                "policy_shuffle")
    assert STAGE2_ARMS["s1_original"]["mode"] == "original_no_oracle"
    assert STAGE2_ARMS["s1_compact"]["mode"] == "original_no_oracle_compact"
    # compact -> shuffle changes only the correspondence
    compact = dict(STAGE2_ARMS["policy_compact"])
    shuffle = {k: v for k, v in STAGE2_ARMS["policy_shuffle"].items() if k != "shuffle"}
    assert compact == shuffle
    assert STAGE2_ARMS["policy_shuffle"]["shuffle"] is True
    # policy_full is kept for the dilution diagnostic but is not a main arm
    assert "policy_full" in STAGE2_ARMS
    assert "policy_full" not in STAGE2_MAIN_ARMS
    assert STAGE2_ARMS["policy_full"]["representation"] == "full"


def test_neither_s1_arm_declares_a_future():
    for arm in ("s1_original", "s1_compact"):
        spec = STAGE2_ARMS[arm]
        assert "continuation" not in spec
        assert "representation" not in spec
        assert not spec.get("shuffle")


@pytest.mark.parametrize("arm", STAGE2_MAIN_ARMS)
def test_every_main_arm_has_a_mode(arm):
    from jev_libero.experiment import MODES

    assert STAGE2_ARMS[arm]["mode"] in MODES
