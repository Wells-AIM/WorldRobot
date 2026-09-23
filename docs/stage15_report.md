# Stage 1.5 — How should a counterfactual future be constructed?

Stage-1 asked how far an embodied agent should reason ahead. Its answer was that
the horizon was not the binding variable, and that the way the future was
*constructed* was. Stage 1.5 changes the question:

> **How should counterfactual futures be constructed under receding-horizon
> embodied control — and does a correctly constructed multi-step future add
> anything beyond a strong one-step consequence baseline?**

Still training-free. Jev frozen, world model frozen, candidate generator frozen.
MuJoCo remains an **oracle world model** with zero model error by construction,
so what is under study is **future-construction error**, not prediction error.

## 1. Why Stage 1.5 was needed

Real control runs `execute a_t → observe s_t+1 → replan`. Stage-1's long futures
did not:

| chunk | what it shows | measured problem |
|---|---|---|
| `repeat` = `[a, a, a]` | the input performed three times | saturates for large steps |
| `hold` = `[a, hold, hold]` | the input, then settling | duplicates the one-step field |

**Saturation, measured over a Stage-1 episode.** 3-step approach relative to the
1-step figure: **2.67×** for 3 mm inputs, **1.17×** for 10 mm, **1.04×** for
40 mm, the 40 mm mean going negative as the chunk overshoots and returns. A
40 mm step beats a 10 mm step by **2.66×** as the single primitive that executes
but by only **1.03×** as a 3-repeat chunk — the representation erases the
distinction the critic needs.

**Redundancy, measured the same way.** With `hold`, the spread of the trajectory
endpoint across candidates matched the spread of the one-primitive field at a
ratio of **1.02** far from the goal and **0.97** near it. It added prompt volume
and no information.

Neither chunk answers: *if `a_t` runs now and the robot then continues sensibly
from the state that produces, what happens?*

## 2. Continuation Policy

`src/jev_libero/continuation.py` adds a pluggable interface:

```python
class ContinuationPolicy:
    def choose_next_action(self, world, first_input, grip, step_index, task_config):
        ...  # -> (action_name, reason)
```

Three implementations: `repeat`, `hold`, `policy_consistent`. `jev_continuation`
and `learned_policy_continuation` are deliberately **not** implemented.

### How `policy_consistent` chooses

At every branch state, inside the rollout:

1. **Offer a narrow continuation set** derived only from the first input's own
   shape — same axis and sign, at magnitudes **at or below** its own, plus
   `hold`. `x+40mm` → `[x+40mm, x+10mm, x+3mm, hold]`; `x+3mm` → `[x+3mm, hold]`;
   a rotation or gripper input → `[itself, hold]`. It can back off or stop, never
   escalate.
2. **Preview each** with the same `World.execute` live execution uses, from a
   `Snapshot` of that branch state.
3. **Apply the task's collision rules** (`hard_feasible`) — the project's own
   `reject_new_obstacles` / `require_obstacle_free_endpoint`.
4. **Evaluate the task's effect contracts** with the success predicate
   neutralised, and take the first contract in the task's declared order that
   any option satisfies.
5. **Rank within that pool** by the task's own `search.score` with its
   `after.success` term removed. Ties break on the fixed `ACTIONS` order.

### The oracle split this required

The bundled tasks' `advance_target` contract reads:

```json
{"any": [{"ge": [{"ref": "p.closing_mm"}, {"ref": "thresholds.progress"}]},
         {"ref": "after.success"}]}
```

It is *partly* the LIBERO success predicate. `neutralize_oracle` forces the
predicate false before the contracts are consulted, keeping the physical branch
and dropping the oracle branch — a split the contracts themselves do not make.
`search.score[0]` is `after.success` and is dropped from the ranking. No reward,
no success predicate, no oracle score reaches a continuation decision.

### Proof that it reads the branch state

Every continuation choice logs its rule, the contract that fired, and the
feasible set it saw, to `continuation.jsonl`. Scanned across four live states and
ten candidates each, **23 of 40 candidate/state pairs stop looking like
`[a, a, a]`**.

Two states, all three continuations, the surface gap in mm at each checkpoint.

**Start state**, 93.48 mm from the drawer:

| first input | repeat | policy_consistent | Δ end gap |
|---|---|---|---:|
| `x+40mm` | 93.5 → 64.4 → 48.7 → 49.5 | 93.5 → 64.4 → 48.7 → 49.5 | 0.00 |
| `x+10mm` | 93.5 → 85.6 → 77.9 → 70.7 | identical | 0.00 |
| `x+3mm` | 93.5 → 91.1 → 88.7 → 86.4 | identical | 0.00 |
| `x-40mm` | 93.5 → 126.8 → 164.6 → **203.1** | 93.5 → 126.8 → 128.9 → **128.6** | **−74.58** |
| `y-40mm` | 93.5 → 93.5 → 95.5 → 93.7 | 93.5 → 93.5 → 93.5 → 93.6 | −0.11 |

**Mid-episode**, 43.2 mm out, after `x+40mm z-40mm y-40mm y-40mm`:

| first input | repeat | policy_consistent | policy actions | Δ end gap |
|---|---|---|---|---:|
| `x+40mm` | 43.2 → **12.2** → 20.9 → **28.7** | 43.2 → 12.2 → 12.4 → **12.6** | `x+40mm, x+10mm, x+10mm` | **−16.06** |
| `x+10mm` | 43.2 → 34.3 → 25.7 → 18.2 | identical | `x+10mm ×3` | 0.00 |
| `x-40mm` | 43.2 → 79.4 → 118.4 → **155.3** | 43.2 → 79.4 → 82.4 → **82.1** | `x-40mm, hold, hold` | **−73.23** |

Two distinct corrections, and it matters which is which:

- **Retreats stop being catastrophic.** `x-40mm` under `repeat` runs away to
  203 mm; the policy holds after one step and ends at 128 mm. A replanning agent
  would never walk backwards three times, so `repeat` was slandering the action.
- **Large approaches stop overshooting** — but only where overshoot is possible.
  At 43.2 mm out, `repeat` shows `x+40mm` ending at **28.7 mm** when the single
  primitive that executes reaches **12.2 mm**: the best available action looks
  2.3× worse than it is. The policy backs off to `x+10mm` and ends at 12.6 mm.

At the start state, 93 mm out, the policy **agrees** with `repeat` for every
approaching action, and that is correct rather than a failure: continuing really
is the best of the offered options there. Measured at the branch state after two
`x+40mm` steps, every option increased the gap and `x+40mm` increased it least
(−0.85 mm against `hold` at −2.00 mm), because the arm is still drifting from the
previous command.

**So the saturation fix is state-dependent.** It appears where overshoot is
physically available, not everywhere. Stage-1's saturation statistic was measured
across a whole episode; this correction applies to the part of it near the goal.

## 3. Code changes

```
 src/jev_libero/cli.py                 |  17 ++-
 src/jev_libero/continuation.py        | 193 ++++++++++++++++++++++++   NEW
 src/jev_libero/experiment.py          | 233 ++++++++++++++++++++++++++++-
 src/jev_libero/futures.py             |  63 ++++++--
 src/jev_libero/runner.py              |  89 +++++++++++-
 tests/test_continuation.py            | 236 ++++++++++++++++++++++++++++++   NEW
 tests/test_continuation_simulation.py | 266 ++++++++++++++++++++++++++++++++++   NEW
 tests/test_futures.py                 |   3 +-
 tools/run_stage15.py                  | 184 +++++++++++++++++++++++   NEW
 9 files changed, 1259 insertions(+), 25 deletions(-)
```

Also new: `tools/continuation_diag.py`, which produced the tables in section 5.

**New:** `continuation.py` (the policy abstraction, the oracle split, the
continuation action set), `test_continuation.py`, `test_continuation_simulation.py`,
`run_stage15.py`, `continuation_diag.py`.

**Modified:** `futures.py` — `rollout` now takes a continuation policy and decides
one primitive at a time, recording what it took and why; `experiment.py` — the
`strong_policy_future` mode, the compact representation, estimated token
accounting, the shuffle wrapper, the novelty diagnostic; `runner.py` — the new
mode, per-decision phases, termination diagnostics, a 60-decision default;
`cli.py` — `--future-representation`, `--shuffle-future-arm`.

Nothing was refactored for its own sake and nothing from Stage-1 was removed.

## 4. Baseline preservation

`original` and `original_no_oracle` are untouched.

- `test_original_mode_is_untouched` still replays the published episode: 20
  decisions, 155 environment steps, success, every recorded API request matching
  field by field.
- `test_strong_baseline_is_not_disturbed` (new) runs `original_no_oracle` and
  asserts it writes no `continuation.jsonl`, reports `chunk_continuation` and
  `future_representation` as `None`, carries **zero** estimated future tokens, and
  sends no `future_trajectory` in any option.
- `strong_policy_future` routes through the *same* `OracleFilteringAPI` as
  `original_no_oracle`, so the trajectory is the only difference between S1 and
  the S1+Future arms.

Measured across the six mock arms, the first-step candidate set is **identical at
every decision** — 10/10 decisions, all six arms.

## 5. Continuation diagnostics

See section 2 — the tables there are the continuation diagnostics, produced by
`tools/continuation_diag.py` at two live states across 40 / 10 / 3 mm scales.

In the real episodes, `continuation.jsonl` records every candidate's simulated
actions and the rule behind each choice. Over the mock six-arm run, chunks that
stop looking like `[a, a, a]`: **`repeat` 0/270**, **`hold` 260/270** (by
construction), **`policy_consistent` 142/270** — the policy re-decided on 53% of
candidate/step pairs and agreed with continuing on the rest.

## 6. Full vs Compact

`PolicyFull` and `PolicyCompact` run **the same continuation policy over the same
rollouts**. `test_full_and_compact_describe_the_same_physics` asserts identical
simulated actions and identical step counts; only the formatting differs.

| | estimated future tokens/decision | reported provider tokens/decision |
|---|---:|---:|
| S1 (no trajectory) | 0 | 1,506 |
| S1+Repeat (full) | 1,930 | 6,003 |
| S1+Hold (full) | 1,684 | 5,676 |
| S1+PolicyFull (full) | 2,128 | 6,582 |
| **S1+PolicyCompact** | **494** | **2,701** |
| S1+PolicyShuffle (compact) | 674 | 3,316 |

Compact carries **4.3× fewer estimated future tokens** than full and **2.4× fewer
reported tokens per decision overall**. The token split is a character-based
estimate and is labelled `estimated` everywhere it appears; the provider returns
only a total, and the ratio between arms is the usable part.

**Full** repeats absolute state at every checkpoint — end-effector pose, finger
gap, contact flags and modes, forces, surface gap, task features, three times
over. **Compact** reports each quantity as its path and each contact fact as its
transitions:

```json
{"horizon_s": 1.2, "continuation": "policy_consistent",
 "t_s": [0.0, 0.4, 0.8, 1.2],
 "paths": {"distance_to_moving_geometry_mm": [43.2, 12.19, 12.39, 12.61],
           "remaining_open_mm": [43.2, 41.9, 38.4, 35.1]},
 "target_contact": [false, false, true, true]}
```

Nothing forbidden appears in either: `contains_oracle` returns empty for both,
asserted on real rollouts.

## 7. Test results

**73 offline** (`tests/`, no simulator, no network) and **15 simulation**
(`--simulation`) tests pass. Stage-1 had 59 + 15; Stage 1.5 adds 14 offline and
15 simulation tests, of which the load-bearing ones are:

| test | what it pins down |
|---|---|
| `diverges_from_repeat_across_the_candidate_set` | the policy is not a disguised repeat |
| `stops_a_retreat_instead_of_repeating_it` | same first step, genuinely different futures |
| `is_stateful_across_steps` | each choice is recorded per step with its own feasible set |
| `is_deterministic` | same snapshot/candidate/horizon → same actions, checkpoints, reasons |
| `does_not_contaminate_the_live_environment` | isolation still bit-exact (`atol=0.0`) with nested rollouts |
| `full_and_compact_describe_the_same_physics` | identical simulated actions and step counts, different formatting only |
| `compact_uses_fewer_tokens_on_real_rollouts` | compact is strictly smaller |
| `neutralize_oracle_forces_the_predicate_false` | the `advance_target` oracle branch is split off |
| `contract_scoring_survives_a_poisoned_success_field` | a raising `success` key is never read |
| `every_stage15_arm_shares_the_first_step_candidate_set` | candidate parity across all six arms |
| `receding_horizon_still_executes_one_primitive` | depth-3 chunks, one primitive live |
| `strong_baseline_is_not_disturbed` | S1 unchanged |

**Known upstream failures, unchanged from Stage-1:** six replay tests and
`test_runner_without_video_or_network` fail on this host at ~6e-12 float drift.
Verified to fail identically on an unmodified upstream checkout — a hardware/BLAS
determinism difference, not a regression.

**Leakage audit on the real request logs.** Across all six mock arms, oracle hits
in the `state` and `criteria` that Jev ranks: **0**. Thirteen hits per arm appear
in the *instruction prose*, all from the original pipeline's own phrase "task
progress". That is generic objective framing rather than a per-candidate verdict,
and rewriting it would move a second variable, so it is left alone and reported
here rather than silently filtered.

## 8. Real Jev results

`top_drawer`, seed 1, initial state 0, K=27, D=3, horizon 1.2 s, cap 60,
TypeSafe `jev-latest`. Every arm shares task, seed, initial state, candidate
generator, primitives, hard feasibility rules, provider and cap.

| Arm | Success | Decisions | Far | Mid | Near | Terminal | Tokens/dec | Rollout ms | Cost |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S1** | ✅ | **32** | 9 | 16 | 6 | 1 | 1,506 | 0 | $0.0020 |
| S1+Repeat | ✅ | 51 | 15 | 20 | 13 | 3 | 6,003 | 706,214 | $0.0129 |
| S1+Hold | ❌ | 40 | 10 | 16 | 14 | 0 | 5,676 | 554,190 | $0.0095 |
| S1+PolicyFull | ✅ | 45 | 10 | 22 | 1 | **12** | 6,582 | 1,626,646 | $0.0124 |
| **S1+PolicyCompact** | ✅ | **26** | 9 | 14 | **1** | 2 | 2,701 | 943,331 | $0.0030 |
| S1+PolicyShuffle | ✅ | 33 | 18 | 11 | 2 | 2 | 3,316 | 1,027,411 | $0.0046 |

Final remaining: S1 −0.06 mm, Repeat −1.68, Hold **+2.61** (unfinished),
PolicyFull −0.57, PolicyCompact −0.41, Shuffle −1.12.

`S1+Hold` terminated on `no_feasible_action` at decision 40, not on the cap: the
pipeline reached a state where no contract passed and the two-step witnesses
could not rescue it.

Two arms had to be re-run: a provider HTTP 520 ended `S1+Repeat` at decision 6
and `S1+PolicyFull` at decision 13 in the first pass. The client now retries
transient 5xx (bounded, backed off, logged) and both completed on the re-run.
Total spend for this section: **$0.044**.

## 9. Phase analysis

Mean physical progress, mm per decision, by band:

| arm | far >50 | mid 20–50 | near 5–20 | terminal <5 |
|---|---:|---:|---:|---:|
| S1 | 12.30 | 1.32 | 2.87 | 2.64 |
| S1+Repeat | 6.80 | 1.49 | 1.25 | 1.77 |
| S1+Hold | 10.26 | 1.93 | 1.11 | 0.00 |
| S1+PolicyFull | 10.26 | 1.64 | **8.73** | **0.40** |
| S1+PolicyCompact | 12.30 | 2.46 | **4.51** | 1.24 |
| S1+PolicyShuffle | 6.02 | 2.50 | 6.68 | 1.76 |

Action size mix, 40 / 10 / 3 mm per band:

| arm | far | mid | near | terminal |
|---|---|---|---|---|
| S1 | 9/0/0 | 1/6/9 | 0/6/0 | 0/1/0 |
| S1+Repeat | 7/7/1 | 1/8/8 | 3/5/5 | 2/1/0 |
| S1+Hold | 7/3/0 | 7/4/4 | 1/0/8 | 0/0/0 |
| S1+PolicyFull | 7/3/0 | 3/7/12 | 0/1/0 | 1/2/7 |
| S1+PolicyCompact | 9/0/0 | 2/2/7 | 0/1/0 | 0/1/0 |
| S1+PolicyShuffle | 5/12/1 | 2/2/5 | 0/1/0 | 1/0/1 |

**The near band is where the policy-consistent future pays.** S1 spends 6
decisions between 5 and 20 mm at 2.87 mm each. Both policy arms spend **1**, at
8.73 and 4.51 mm per decision. `repeat` and `hold` spend 13 and 14 there at
about 1.2 mm each — `hold` never leaves it.

**The dilution cost moved, it did not disappear.** Stage-1 located the slowdown
at 5–20 mm. With a policy-consistent future the near band is solved, and
`PolicyFull` then loses **12 decisions below 5 mm at 0.40 mm each** while
`PolicyCompact` spends 2. Same physics, same continuation, 4.3× the future
tokens: the grinding reappears at the finest scale, where the prompt is longest
relative to the decision being made.

**`S1+PolicyCompact` keeps the baseline's far-band behaviour** — 9 decisions at
12.30 mm, action mix 9/0/0, identical to S1. `Shuffle` does not: 18 far-band
decisions at 6.02 mm with a 5/12/1 mix, trading large steps for medium ones.

## 10. Interpretation

### Measured facts

Single episode per arm, one task, one seed, one initial state.

1. **Decisions to success:** PolicyCompact **26** < S1 **32** ≈ Shuffle **33** <
   Hold **failed at 40** < PolicyFull **45** < Repeat **51**.
2. **Every full-representation arm is worse than the baseline**, whatever
   continuation built it: 51, 45, and one failure against S1's 32.
3. **Both compact arms are at or better than the baseline**: 26 and 33.
4. **PolicyFull and PolicyCompact ran identical physics.** Same continuation
   policy, same rollouts, asserted by test. They differ only in formatting, and
   they differ by **19 decisions** and 2.4× reported tokens.
5. **The near band (5–20 mm) is where policy-consistent futures pay:** 1 decision
   for both policy arms against S1's 6, at 8.73 and 4.51 mm per decision against
   2.87.
6. **The full representation's cost lands below 5 mm:** PolicyFull spends 12
   decisions there at 0.40 mm each; PolicyCompact spends 2.
7. **The continuation policy re-decides**, 142/270 candidate/step pairs in the
   mock run, and its corrections are physically legible: retreats are held rather
   than repeated (`x-40mm` ends at 128 mm instead of 203 mm), and large
   approaches back off before overshooting (`x+40mm` ends at 12.6 mm instead of
   28.7 mm where the executed step reaches 12.2 mm).
8. **Compact is 4.3× smaller** in estimated future tokens and carries no oracle.

### Interpretation — labelled as such

**Representation appears to dominate construction.** Fact 4 is the cleanest
comparison in the experiment, because the physics is held identical by
construction and by test. The 19-decision gap is therefore attributable to the
prompt, not the future. If that holds up under repetition, prompt dilution is a
larger effect than anything Stage-1 measured about horizons.

**The correspondence effect is real but smaller, and not cleanly separated.**
PolicyCompact 26 against Shuffle 33 is a 7-decision gap with the same schema and
comparable token volume, which is the signature the negative control was built to
detect. But Shuffle ≈ S1, so a deranged future neither helps nor hurts relative
to having no future at all. On a single episode, 26 vs 32 vs 33 are margins that
one lucky rollout could produce.

**A caution against reading fact 5 too strongly.** Shuffle also does well in the
near band (2 decisions, 6.68 mm each), and its future is deliberately wrong.
That weakens "the *right* future fixes the near band" and is more consistent with
"a compact prompt fixes the near band". These cannot be separated with n=1.

**What this does not show.** It does not show that multi-step future reasoning is
valuable in general, that 26 < 32 would survive repetition, or that the effect
transfers to other tasks or initial states. Nothing here was repeated.

## 11. Decision for the next stage

Against the cases defined for this stage, the result is closest to **Case A with
a Case C caveat**, and the caveat should drive what happens next.

**Case A holds on the headline number:** `PolicyCompact` reaches success in 26
decisions against the baseline's 32, with fewer near-band decisions and the same
far-band behaviour. **The Case C caveat:** `Shuffle`, whose future is wrong by
construction, lands at 33 — statistically indistinguishable from the baseline on
one episode. So the evidence that Jev is *using the correspondence* is weaker
than the evidence that it is *hurt by long prompts*.

**Recommended next step — repetition before expansion.**

1. **Repeat the six arms across seeds and initial states on `top_drawer`.**
   Three to five seeds would tell us whether 26 < 32 < 33 is an ordering or a
   coincidence. This is the cheapest decisive experiment available: about $0.04
   per six-arm sweep.
2. **Then repeat on `microwave` and `alphabet_soup`** — the other bundled tasks,
   not new ones.
3. **Only after that**, horizon ablation, now against `PolicyCompact` rather than
   against a weak baseline.

**Explicitly not recommended yet:** the 8-task benchmark, perturbation studies,
world-model mismatch, or any horizon sweep. Stage 1.5's own finding is that a
representation change moved the outcome by 19 decisions while holding the physics
fixed; sweeping the horizon before that is understood would sweep the wrong
variable.

**One design question this raises.** If dilution is the dominant effect, the
interesting direction is not longer futures but *gated* ones — showing a
trajectory only where it discriminates, and the one-step fields alone elsewhere.
`future_novelty_count` already measures, per candidate, whether the trajectory
adds anything the first step did not. It is currently diagnostic only and is
never shown to Jev or used for ranking; making it a gate would be a deliberate
next design step, not a silent change.

### Answers to the stage questions

| | question | answer |
|---|---|---|
| Q1 | Does policy-consistent continuation fix repeat's misrepresentation of large actions? | **Partly, and state-dependently.** It corrects retreats everywhere and overshooting approaches where overshoot is physically available; at the start state it agrees with `repeat` because continuing really is best there. |
| Q2 | Does policy future add information over hold future? | **Yes on outcome** — PolicyFull 45 and success, Hold 40 and failure — though both are far worse than compact. |
| Q3 | Is Compact markedly cheaper than Full? | **Yes.** 4.3× fewer estimated future tokens, 2.4× fewer reported tokens per decision. |
| Q4 | Does shortening the prompt improve late-stage efficiency? | **Yes, and this is the largest effect measured.** Identical physics, 45 → 26 decisions, with the saving concentrated below 5 mm. |
| Q5 | Does a correct multi-step future have marginal value over a strong one-step baseline? | **Suggested, not established.** 26 vs 32 on one episode, and the shuffled control sits at 33 rather than clearly worse. |
| Q6 | If not, can we conclude one-step consequence suffices? | **Not yet, and the opposite is not established either.** What can be said is that a *badly represented* future is reliably worse than none. |
