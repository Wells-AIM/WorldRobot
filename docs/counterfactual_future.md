# Counterfactual future reasoning (CF-Jev)

## Research question

> Can explicit reasoning over counterfactual future consequences improve embodied
> action selection — and how far into the future should an embodied agent reason
> before acting?

This stage is entirely **training-free**. Jev is frozen, the world model is
frozen, the candidate generator is frozen. Nothing here trains a VLA, a policy,
or a neural dynamics model.

**MuJoCo is used as an oracle / simulator-based world model. It is NOT a learned
neural world model.** There is no video world model and no learned dynamics.

## What was already here, and what is new

The original `jev-libero` **already performs local physics preview**. Claiming
"we added world modeling" would be wrong. The honest statement is narrower:

| | original | CF-Jev |
|---|---|---|
| branch mechanism | reversible MuJoCo `Snapshot` | the same `Snapshot`, reused |
| breadth | all 27 atomic inputs | K configurable candidates (default 27 = all feasible) |
| depth | 1 primitive (0.4 s) | D primitives to a configurable horizon (default 1.2 s) |
| second step | `two_step_witnesses`, only when no contract passes | always, as part of the trajectory |
| what Jev receives | scalar effect fields of the end state, plus a two-step witness | timed checkpoints and physical events across the trajectory |
| success predicate | shown to Jev (`predicted_task_complete`, `can_finish_task`) | withheld |

The core extension is **trajectory-level, multi-step, counterfactual future
reasoning**, plus the removal of the simulator's own verdict from the prompt.

## Original architecture (real call chain)

```
runner.run
 └─ World(init_index, seed, task_config)                       world.py:60
    loop:
    ├─ World.predict_all(grip)                                 world.py:243
    │   ├─ Snapshot(self)                                      world.py:16
    │   └─ for each of 27 ACTIONS: restore → execute → features
    │       └─ World.execute = 8 env steps @ 20 Hz = 0.4 s     world.py:187
    ├─ World.two_step_witnesses(grip)  [only if no contract]   world.py:270
    ├─ policy.contracts(...)          task effect criteria     policy.py:25
    ├─ ValidatedPolicy.choose                                  policy.py:110
    │   ├─ api.choose("intent",   ...)
    │   ├─ api.choose("strategy", ...)  → policy.family        policy.py:67
    │   └─ api.choose("motor",    ...)  → one atomic input
    ├─ World.execute(choice)           one primitive only
    └─ records: predictions.jsonl, trace.jsonl, metrics.jsonl
```

Two properties of the original engine matter here and are preserved:

1. `Snapshot.restore()` is **bit-exact**. The runner asserts
   `prediction_state_max_error <= 1e-7` on every decision and raises otherwise.
2. The engine **already executes only one primitive per decision**. CF-Jev keeps
   that; the chunk is a reasoning object, not a plan to run.

## Experiment modes

All modes share the task, the initial state, the seed, the candidate generator,
the candidate count, the action primitives, the hard feasibility rules, the Jev
version, the provider, and the decision budget. The **only** core difference is
how much future consequence information the critic receives.

| mode | horizon | what the critic sees |
|---|---|---|
| `original` | 0.4 s (+ 2-step witness) | the published three-layer pipeline, untouched |
| `original_no_oracle` | same | the same pipeline with the success predicate filtered out |
| `original_deep` | D primitives | the same pipeline previewing a chunk endpoint instead of one primitive |
| `original_trajectory` | D primitives | the same pipeline, one-primitive fields kept, plus trajectory checkpoints |
| `reactive` | none | task, current state, candidate list |
| `short_preview` | 0.4 s | one primitive of consequence per candidate |
| `counterfactual_future` | configurable, default 1.2 s | trajectory checkpoints and events |
| `shuffle_future` | same as CF | the same futures, deranged against their candidates |

`reactive` → `short_preview` → `counterfactual_future` is a pure horizon sweep.
`shuffle_future` is the negative control.

### Why `original` is separate

`original` is the **system baseline** and is byte-for-byte the published
pipeline: three Jev layers, task effect contracts, two-step witnesses, and the
original prompts including `predicted_task_complete`. It is kept so published
results stay reproducible. The four controlled arms are a separate, stricter
comparison and do not touch it.

## Candidate fairness

Candidates are generated once per decision, from state alone, by
`futures.generate_candidates`:

1. `World.predict_all` previews all 27 inputs (the original code path).
2. `futures.hard_feasible` applies the task's **collision rules only**
   (`reject_new_obstacles`, `require_obstacle_free_endpoint`). Rejections are
   recorded per decision in `hard_rejected`.
3. Surviving inputs are grouped by `policy.family` (contact mode × motion kind).
4. K slots are allocated across families by largest remainder **in proportion to
   family size**, and picks are spread across each family including its
   endpoints. No score, no reward, no success predicate is consulted.
5. Each first input is extended to depth D by the configured continuation:
   `repeat` gives `[a, a, a]`, `hold` gives `[a, hold, hold]`. See
   `--chunk-continuation`; the choice is not neutral, and the measured
   consequences are below.

Because selection depends only on the state, the candidate set is identical
across all four arms by construction. This is asserted offline
(`test_candidate_set_is_identical_across_controlled_modes`) and against live
physics (`test_candidate_sets_match_across_modes_on_live_physics`), and a
further test flips the success predicate on every prediction and shows the
candidate set does not move.

### Why K defaults to 27

The default offers **every feasible input**, which removes the sampler from the
experiment entirely. That matters more than it looks, and the reason is worth
recording.

The first implementation took candidates round-robin across families. At the
`top_drawer` start state the feasible pool is 18 translations, 6 rotations,
2 finger inputs and 1 hold, so round-robin returned a six-item menu holding a
**single** translation — while every large approach move lives in the
translation family. The one translation that made the cut, `x-40mm`, retreats
from the target. A real Jev run then made ten decisions and zero task progress,
not because future reasoning failed but because no useful action was ever
offered.

Proportional allocation fixes the shape of the menu, but measurement at the same
state shows K=6 is still too small to cover 27 inputs:

| K | best approach on the menu | top-5 approach moves offered |
|---|---|---|
| 6 | 0.14 mm | none |
| 8 | 7.89 mm | 2 of 5 |
| 12 | 29.11 mm | 3 of 5 |
| 27 | 29.11 mm | all |

A menu that rarely contains a good action floors every arm at zero success and
leaves the comparison with no headroom — the experiment would measure nothing.
K=27 costs 27 × D × 8 environment steps per decision (648 at D=3, about 10.7 s
on CPU) and is linear in K, not the forbidden exponential `27^D` search.

Smaller K remains available via `--candidate-count` for cost-constrained sweeps,
and `allocate`/`spread` keep the menu proportional at any K.

**Limitation.** Neither continuation is a search over action sequences, and
mixed-primitive chunks are unexplored. `repeat` asks "what if the robot keeps
doing this?", which systematically misrepresents large steps; `hold` asks "what
if the robot does this and settles?", which turns out to duplicate the
one-primitive field. The interface takes any list of primitives, so a real
proposer can be dropped in without touching the rollout or summary code.

## Counterfactual rollout

`futures.rollout_all` takes one `Snapshot` of the live state and reuses it for
every candidate:

```python
root = Snapshot(world)
for candidate in candidates:
    root.restore()          # every candidate starts from the same state
    rollouts[...] = rollout(world, candidate, grip, horizon, cfg, snapshot=root)
root.restore()              # live env restored, in a finally block
```

Each `rollout` runs `World.execute` — the same function live execution uses — so
a branch is the same physics as reality, not an approximation.

Isolation is verified at `atol=0.0`: `qpos`, `qvel`, the flattened sim state and
`data.time` come back **bit-identical** after a rollout
(`test_rollout_does_not_contaminate_the_live_environment`).

## Future summary schema

The world model reports objective physical consequences and nothing else.
`futures.ORACLE_MARKERS` bans any key or string containing `success`, `reward`,
`task_score`, `oracle`, `best_action`, `good_action`, `best_candidate`,
`recommended`, `can_finish`, `complete`, `progress`, or `verdict`.

Per checkpoint: `eef_mm`, `finger_gap_mm`, `moving_contact`, `obstacle_contact`,
`contact_modes`, `target_force_N`, `obstacle_force_N`,
`distance_to_moving_geometry_mm`, plus the task's own declared `features`
(for example `qpos_m` and `remaining_open_mm`) minus anything matching a marker.
Nothing is invented: these come from the existing `measurements` / `features`
system.

Events: `non_target_contact` (with the geom pairs), `target_contact_lost`,
`target_contact_established`.

A real generated example (`top_drawer`, seed 1, init 0, step 0, candidate C1):

```json
{
  "input": "Open gripper fingers; hold position and orientation.",
  "motion_family": "no_target_contact__finger",
  "planned_chunk": ["open", "open", "open"],
  "future": {
    "horizon_s": 1.2,
    "simulated_steps": 24,
    "checkpoints": [
      {"t_s": 0.0, "after_primitive": 0, "input": null,
       "eef_mm": [-207.912, -14.459, 1178.262], "finger_gap_mm": 76.446,
       "moving_contact": false, "obstacle_contact": false, "contact_modes": [],
       "target_force_N": 0, "obstacle_force_N": 0,
       "distance_to_moving_geometry_mm": 93.484,
       "qpos_m": -0.15166, "remaining_open_mm": 151.659},
      {"t_s": 0.4, "after_primitive": 1, "input": "open", "...": "..."},
      {"t_s": 0.8, "after_primitive": 2, "input": "open", "...": "..."},
      {"t_s": 1.2, "after_primitive": 3, "input": "open", "...": "..."}
    ],
    "events": [],
    "net_eef_displacement_mm": 0.118,
    "ends_in_target_contact": false,
    "ends_in_non_target_contact": false,
    "peak_non_target_force_N": 0.0,
    "event_count": 0,
    "net_surface_gap_change_mm": -0.219,
    "net_qpos_m_change": 0.0,
    "net_remaining_open_mm_change": 0.0
  }
}
```

There is no "this is the best candidate" field anywhere. Judging the future is
Jev's job.

### A leak the controlled arms had to close

The task configuration's `feedback` projection for `top_drawer` declares a
`success` field, so `recent_real_outcomes` put the LIBERO predicate in front of
the critic on every decision after the first. Step-0 tests could not see it.
`experiment.decide` now filters history entries through the same marker list,
and `test_recent_outcomes_are_stripped_of_the_success_predicate` covers it.

## Receding horizon

CF-Jev reasons over `[a_t, a_t+1, a_t+2]` but the live environment only ever
receives `a_t`:

```
plan chunk → execute first primitive only → observe → recompute candidates
           → re-simulate futures → ask Jev again
```

The chunk is never executed as a plan. Re-planning after every primitive is what
makes the comparison about *reasoning depth* rather than *open-loop plan
length*, and it keeps every arm on the same environment-step budget:
8 decisions × one 8-step primitive = 64 environment steps in every mode.

## Horizon is configurable

`horizon_for(mode, requested)` maps `reactive → 0.0` and
`short_preview → 0.4`; the counterfactual arms take the requested value.
`0`, `0.4`, `0.8`, `1.2`, and `2.0` seconds are all supported and nothing is
hard-coded to 1.2. `test_horizon_controls_how_far_the_future_reaches` asserts
that 0.4/0.8/1.2 s simulate 1/2/3 primitives respectively.

Each decision logs `future_horizon_requested_s`, `future_horizon_actual_s`, and
`simulated_steps`, because a primitive can terminate early and the actual
horizon then falls short of the request.

### Compute cost vs future horizon (measured)

Ten decisions per cell, K=6, chunk depth matched to the horizon, mock provider,
CPU. `tools/` sweep across all three bundled tasks:

| horizon (s) | rollout steps | rollout ms / decision |
|---:|---:|---:|
| 0.0 | 0 | 0 |
| 0.4 | 480 | ~775 |
| 0.8 | 960 | ~1560 |
| 1.2 | 1440 | ~2310 |
| 2.0 | 2400 | ~3840 |

Cost is linear in the horizon at about **1950 ms per simulated second** per
decision, consistent across `microwave`, `top_drawer` and `alphabet_soup`.
Executed environment steps stayed at 80 in every cell, so the live budget is
controlled while only the counterfactual cost moves.

At the K=27 default the same relationship holds with the constant scaled by
27/6: 648 rollout steps and ~10.7 s per decision at a 1.2 s horizon.

API cost scales with K too. Measured on TypeSafe at K=27, D=3, 1.2 s horizon:
**32,385 input tokens per decision**, about $0.00136 per decision at the
published $0.042/M input price.

## Testing the published pipeline against itself

The four controlled arms measure future information against a baseline that had
been stripped of the published pipeline's three decision layers, its effect
contracts and its task-aligned derived fields. That answers a question about the
stripped setup, not about the real system. Two variants change exactly one thing
about the published pipeline instead.

`original_no_oracle` routes the provider through `OracleFilteringAPI`, which
drops `ORACLE_MARKERS` keys from the state and criteria at the transport
boundary. `predicted_task_complete` and `can_finish_task` disappear;
`predicted_closing_mm`, the surface-gap reduction and the contact flags stay.
Instruction text is left alone — the phrase "task progress" there frames the
objective rather than ranking candidates, and rewriting it would move a second
variable.

`original_deep` previews each input over D repetitions through
`World.predict_all_deep`. The prediction and policy projections are declarative
over `before`/`after`, so the prompt structure, the contracts and the layered
decisions carry over untouched and only the horizon moves. The shallow
predictions are kept for the runner's execution-match assertion, since one
primitive still executes.

### Results, original family, one task / seed / initial state

`top_drawer`, seed 1, init 0, TypeSafe `jev-latest`.

| arm | decision cap | success | decisions | tokens / decision | cost |
|---|---:|:---:|---:|---:|---:|
| original, verdict shown | 30 | yes | **17** | 1,827 | $0.0013 |
| original, verdict withheld | 30 | yes | **25** | 1,541 | $0.0016 |
| original_deep D=3 | 60 | yes | **49** | 2,024 | $0.0042 |
| original_trajectory D=3, repeat | 60 | yes | **49** | 7,116 | $0.0146 |
| original_trajectory D=3, hold | 60 | yes | **49** | 5,459 | $0.0110 |

**Every variant succeeds.** Adding future information to the published pipeline
does not break it; it makes it slower, by a factor of about three. All three
augmented variants land on exactly the same 49 decisions, whether the path is
discarded (`original_deep`), preserved (`original_trajectory` with `repeat`), or
preserved with a settling continuation (`hold`) — the last of which fixes the
early phase and still finishes at 49. Withholding the success predicate costs
about eight decisions. The next section takes this apart.

#### Two earlier conclusions were wrong, and why

This section previously reported that the deep variants *failed* and concluded
that "naive horizon extension hurts" and that "the endpoint alone is worse than
no extension". Both were artifacts of the measurement, not findings:

1. **A witness-gate bug.** The two-step escape hatch was gated on
   `mode == "original"`, so the variants ran without it while their baseline had
   it. `original_no_oracle` failed on `no_feasible_action` because of that, not
   because of the missing predicate. Fixed, then re-measured.
2. **A decision cap set from the baseline.** The cap was 30 because the original
   needs 17. The variants were still descending when it hit — `original_deep`
   at 46.4 mm remaining, `original_trajectory` at 33.3 mm — and both reach the
   goal at decision 49 when the cap is 60. What looked like a plateau was the
   budget running out.

The general lesson is recorded here because it applies to the controlled arms
too: **a decision cap chosen from the fastest arm turns "slower" into "failed".**
Any arm that is still making monotone progress when the cap hits has not been
measured, only truncated.

### The controlled arms have the same problem

The four-arm comparison in this document was run at the same 30-decision cap.
`reactive`, `short_preview` and `shuffle_future` made no progress at all, so the
cap is unlikely to matter for them. **`counterfactual_future` did not plateau**:
it was at 40.4 mm and still descending at decision 30. Its reported non-success
is therefore a truncation, and the four-arm table below must be re-run at a
higher cap before its numbers are read as outcomes rather than as progress
snapshots. The contrast it shows against the other three arms — 111 mm closed
and 18 of 30 states in contact, against 0 mm and no contact — does not depend on
the cap.

### Four controlled arms (30-decision cap; see the caveat above)

| | reactive | short_preview | counterfactual | shuffle |
|---|---:|---:|---:|---:|
| horizon | 0 s | 0.4 s | 1.2 s | 1.2 s (deranged) |
| decisions / env steps / API calls | 30 / 240 / 30 | 30 / 240 / 30 | 30 / 240 / 30 | 30 / 240 / 30 |
| input tokens per decision | 1,970 | 18,725 | 31,771 | 31,142 |
| drawer closed | 0.00 mm | 0.00 mm | 111.25 mm | 0.00 mm |
| states in contact with target | 0 / 30 | 0 / 30 | 18 / 30 | 0 / 30 |
| final surface gap | 637.80 mm | 306.69 mm | 0.00 mm | 493.22 mm |
| still descending at the cap | no | no | **yes** | no |
| API cost | $0.0025 | $0.0236 | $0.0400 | $0.0392 |
| oracle leakage | 0 | 0 | 0 | 0 |

The two controls still do their job. `short_preview` at 0.4 s behaved exactly
like no future at all, so the difference is not the mere presence of a predicted
outcome. `shuffle_future` carries 31,142 input tokens per decision against the
counterfactual arm's 31,771 — within 2%, same schema, same candidate set, zero
derangement fixed points — and shows none of the progress, so the difference is
not prompt length. What tracks it is the correspondence between an action and
its own future.

### Why the augmented arms are slower

Measured from the episode logs, no extra API spend.

The augmented arms pick small increments where the original picks large ones.
Translations by size, over the whole episode:

| arm | 40 mm | 10 mm | 3 mm | early progress (≥50 mm out) |
|---|---:|---:|---:|---:|
| original | 12 | 0 | 2 | **12.30 mm / decision** |
| original_deep D=3 | 10 | 16 | 22 | 3.54 |
| original_trajectory, repeat | 12 | 20 | 17 | 4.95 |
| original_trajectory, hold | 16 | 9 | 17 | **10.26** |

**The chunk, not the horizon, was doing the damage.** A candidate chunk repeats
its own primitive D times, so a 40 mm input is shown as 120 mm of travel — and
that saturates. Across the episode the 3-step approach is 2.67x the 1-step
figure for 3 mm inputs, 1.17x for 10 mm, and 1.04x for 40 mm, the 40 mm mean
going negative as the chunk overshoots and comes back. The consequence is
direct: a 40 mm step beats a 10 mm step by **2.66x** as the single primitive
that executes, but by only **1.03x** as a 3-repeat chunk. The representation
erases the distinction the critic needs for the decision it is making.

Switching the continuation to `hold` — the input, then settle — recovers most of
the early-phase speed, 4.95 to 10.26 mm per decision against the original's
12.30, and the action mix flips back to large steps. **It does not recover the
decision count**: `hold` still finishes in 49 decisions, the same as `repeat`,
because it spends 39 decisions below 50 mm where the original spends 8, and 24
of those grinding between 5 and 20 mm.

A hypothesis for that endgame — that candidate trajectories become
indistinguishable near the goal — was tested and is **false**. Comparing the
spread of the trajectory endpoint across candidates against the spread of the
one-primitive field:

| arm | phase | 1-step spread | trajectory spread | ratio |
|---|---|---:|---:|---:|
| repeat | ≥50 mm | 3.96 | 18.28 | 4.62 |
| repeat | <50 mm | 4.15 | 23.71 | 5.72 |
| hold | ≥50 mm | 6.45 | 6.59 | 1.02 |
| hold | <50 mm | 3.51 | 3.41 | 0.97 |

The trajectory keeps separating the candidates. What the numbers show instead is
that the two continuations fail in different ways: `repeat` carries a large
signal pointing the wrong way, and `hold` carries a signal that is **a near
duplicate of the one-primitive field it sits next to** — ratio 1.0 in both
phases — so it adds prompt volume and no information.

**The late-phase slowdown remains unexplained.** Prompt dilution is the obvious
candidate and has not been tested. It should not be asserted until it is.

### What this says about the research question

Two settings, and they disagree in a way that now makes sense:

- **Given accurate one-primitive consequences of the action that will actually
  run**, reasoning further ahead did not improve selection in any form tried.
  The longer horizon is either about a different action (`repeat`, which shows
  the input done three times) or about nothing new (`hold`, whose endpoint
  duplicates the one-step field). Both cost decisions; neither saves any.
- **Given none of those fields**, trajectory-level future is the difference
  between approaching the drawer and wandering away from it: 111 mm closed and
  18 of 30 states in contact, against 0 mm and no contact for the reactive,
  short-preview and shuffled arms.

So on this evidence the useful question is not how far to reason, but **what the
critic already knows about the step it is about to take**. Counterfactual future
reasoning paid for itself only where that knowledge was missing.

One task, one seed, one initial state, one episode per arm.

## Shuffle future (negative control)

Futures are generated normally, then reassigned by a **derangement** — a
permutation with no fixed point — seeded by `shuffle_seed + step`. Candidate
count, summary schema, field names, checkpoint count, and the request structure
are unchanged; only the action↔future correspondence is broken. Each decision
records `original_future_id`, `assigned_future_id`, and `shuffle_seed`.

If CF-Jev beats shuffle-CF, the gain comes from *which* future belongs to *which*
action, not from the presence of extra tokens.

## CLI

```bash
jev-libero run --task top_drawer --mode reactive \
  --provider mock --seed 1 --init-state 0 --out runs/reactive

jev-libero run --task top_drawer --mode short_preview --future-horizon 0.4 \
  --provider mock --seed 1 --init-state 0 --out runs/short

jev-libero run --task top_drawer --mode counterfactual_future \
  --future-horizon 1.2 --candidate-depth 3 --candidate-count 6 \
  --provider mock --seed 1 --init-state 0 --out runs/cf

jev-libero run --task top_drawer --mode shuffle_future \
  --future-horizon 1.2 --shuffle-seed 0 \
  --provider mock --seed 1 --init-state 0 --out runs/shuffle
```

`--provider mock` is a deterministic offline stand-in: no network, no
credentials, no spend. **It is not Jev**, and runs made with it must never be
reported as Jev results. Swap in `--provider typesafe` or `--provider openrouter`
for real, paid episodes.

Pilot sweep:

```bash
python tools/run_cfjev_pilot.py \
    --tasks microwave top_drawer alphabet_soup \
    --modes reactive short_preview counterfactual_future shuffle_future \
    --init-state-ids 0 --horizon 1.2 --provider mock --out runs/pilot
```

It writes `pilot_summary.csv` and `pilot_summary.json` with task, seed,
`init_state_id`, mode, success, decision count, environment steps, collision and
event counts, rollout steps, latencies, API calls, and cost.

## Logging

Controlled modes write `counterfactual.jsonl`, one row per decision, alongside
the existing records. The original `predictions.jsonl`, `trace.jsonl`,
`metrics.jsonl`, and `summary.json` keep their schema; new keys are added, none
are replaced or removed.

Each row carries `experiment_mode`, requested and actual horizon,
`candidate_depth`, `candidate_count`, `shuffle_seed`, the full candidate list
with `original_future_id` / `assigned_future_id` and the complete
`CounterfactualRollout` (steps, duration, latency, checkpoints, events,
summary), `hard_rejected`, the exact `jev_request`, the selected candidate, and
the executed input. `summary.json` adds `rollout_steps_total` and
`rollout_latency_ms_total`.

## Tests

Offline (`tests/test_futures.py`, no simulator, no network):

- derangement has no fixed point, is deterministic, rejects a single candidate
- `strip_oracle` drops the success predicate; `contains_oracle` finds planted markers
- candidate sets identical across the four modes on recorded physics
- candidate generation is unmoved when the success predicate is flipped
- candidate count and depth are configurable; hard rejections partition the pool
- reactive prompts carry no future and no oracle marker
- recent outcomes are stripped of the success predicate
- horizon mapping per mode

Simulation (`tests/test_counterfactual_simulation.py`, `--simulation`):

- rollouts do not contaminate the live environment (`atol=0.0`)
- every candidate rolls out from the same snapshot
- rollouts are deterministic
- the predicted future matches real execution of the same primitives
- horizon controls simulated depth (0.4/0.8/1.2 s → 1/2/3 primitives)
- candidate sets match across modes on live physics
- no oracle verdict in any simulated prompt
- shuffle deranges the correspondence and keeps the payload shape
- every controlled mode completes a full mock episode
- receding horizon: a depth-3 chunk advances the live env by one primitive
- the original pipeline still reproduces its recorded episode

No test contacts TypeSafe, OpenRouter, or the network. The autouse
`without_live_api_credentials` fixture deletes provider keys from the
environment.

## Known limitations

1. **Depth extension is repetition.** A candidate is one primitive repeated D
   times, not a searched sequence. Mixed-primitive chunks are unexplored.
2. **Cost.** At the K=27 default a CF decision simulates 648 environment steps
   for the futures on top of the 216 the original 27-input preview already
   spends — about 10.7 s of rollout per decision on CPU, and ~32k input tokens
   per API call at a 1.2 s horizon.
3. **n=1 everywhere.** Every arm reported here is a single episode on one task,
   one seed and one initial state. There are no error bars and no repetition, so
   none of it can separate a real effect from one lucky rollout.
4. **Mock results are not Jev results.** Anything reported from
   `--provider mock` validates the pipeline, not the hypothesis. The mock
   chooser is random; its episodes never succeed and must never be read as
   evidence about future reasoning.
5. **The candidate budget is a live confound at small K.** K=27 avoids it, but
   any sweep that lowers K to save compute reintroduces it, and a menu that
   misses the task-relevant actions floors every arm. Check the measured
   coverage table above before trusting a small-K comparison.
6. **`original` still leaks.** The published pipeline shows Jev
   `predicted_task_complete` and `can_finish_task`. It was left untouched for
   reproducibility, so `original` is not leak-comparable with the controlled arms.
7. **Events are generic.** `object_dropped`, `joint_limit`, and `IK failure`
   are not emitted: the current measurement system does not compute them
   reliably for all three tasks, and inventing them would be worse than
   omitting them.
8. **Decision caps are part of the measurement.** A cap chosen from the fastest
   arm reports slower arms as failures. Two conclusions in this document were
   retracted for exactly that reason. Before reading any non-success, check
   whether the arm was still making monotone progress when the cap hit.
9. **`original_deep` reports only the chunk endpoint,** hiding the step that
   actually executes. It still reaches the goal, in the same 49 decisions as the
   path-preserving variant, so on this task the hidden intermediate states cost
   nothing measurable — but the field it shows the critic does not describe what
   will run, and that remains worth knowing when reading its prompts.
10. **Horizon granularity is one primitive.** A horizon between multiples of
   0.4 s truncates to whole primitives.
11. **This host does not reproduce published float determinism.** Six upstream
   replay tests and `test_runner_without_video_or_network` fail here at ~6e-12
   on an *unmodified* checkout. See below.

## Stage-1 scope

Deliberately **not** done here: learned world models, Transformer dynamics,
video world models, NanoJev fine-tuning, VLA/OpenVLA/π0 training, RoboTwin, real
robots, LIBERO-100, mass/friction mismatch, perturbation benchmarks,
mixed-effects statistics, and the 8-task main experiment.

## Next stage

1. **Repetition.** The four-arm pattern has been seen once. Repeat it across
   seeds and initial states on all three bundled tasks before believing it.
2. **Re-run the four controlled arms at a higher cap.** The counterfactual arm
   was still descending at decision 30, so its non-success is a truncation.
   Until that is redone, the four-arm table reports progress, not outcomes.
3. **Explain the endgame.** The early-phase half of the slowdown is diagnosed
   and fixed by the `hold` continuation. The late phase is not: `hold` still
   spends 39 decisions below 50 mm against the original's 8, and the obvious
   hypothesis — that candidate trajectories stop discriminating near the goal —
   was tested and found false. Prompt dilution is the next candidate and has not
   been tested.
4. **Horizon ablation** against the strong baseline, sweeping
   0 / 0.4 / 0.8 / 1.2 / 2.0 s. The interface already supports it.
5. **Future-information ablation**: drop checkpoints, drop events, keep only the
   final state, to find which part of the trajectory carries the signal. The
   deep variant is already one point on that curve, and it is the worst one.
6. **Expand to 8 LIBERO tasks** once the three-task mechanism is stable.
7. **Perturbation / world-model mismatch**: mass and friction offsets between
   the rollout model and the live environment, to test whether reasoning over an
   imperfect world model still helps.
