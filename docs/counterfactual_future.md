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
| breadth | all 27 atomic inputs | K configurable candidates (default 6) |
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
4. K first inputs are taken **round-robin across families** in the fixed
   `ACTIONS` order. No score, no reward, no success predicate is consulted.
5. Each first input is extended to depth D by repetition: `[a, a, a]`.

Because step 5 depends only on the state, the candidate set is identical across
all four arms by construction. This is asserted offline
(`test_candidate_set_is_identical_across_controlled_modes`) and against live
physics (`test_candidate_sets_match_across_modes_on_live_physics`), and a
further test flips the success predicate on every prediction and shows the
candidate set does not move.

**Limitation.** Depth extension by repetition is a deliberate v1 choice: it is
deterministic, reward-free, and asks exactly "what if the robot keeps doing
this?". It is not a search over action sequences, and it does not explore
mixed-primitive chunks. The interface takes any list of primitives, so a richer
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
2. **Cost.** A CF decision at K=6, D=3 simulates 144 environment steps for the
   futures on top of the 216 the original 27-input preview already spends. In
   the pilot this was ~2.3 s of rollout per decision on CPU.
3. **Mock results are not Jev results.** Everything reported here from
   `--provider mock` validates the pipeline, not the hypothesis. Nothing can be
   said about whether counterfactual reasoning helps until real Jev episodes run.
4. **`original` still leaks.** The published pipeline shows Jev
   `predicted_task_complete` and `can_finish_task`. It was left untouched for
   reproducibility, so `original` is not leak-comparable with the controlled arms.
5. **Events are generic.** `object_dropped`, `joint_limit`, and `IK failure`
   are not emitted: the current measurement system does not compute them
   reliably for all three tasks, and inventing them would be worse than
   omitting them.
6. **Horizon granularity is one primitive.** A horizon between multiples of
   0.4 s truncates to whole primitives.
7. **This host does not reproduce published float determinism.** Six upstream
   replay tests and `test_runner_without_video_or_network` fail here at ~6e-12
   on an *unmodified* checkout. See below.

## Stage-1 scope

Deliberately **not** done here: learned world models, Transformer dynamics,
video world models, NanoJev fine-tuning, VLA/OpenVLA/π0 training, RoboTwin, real
robots, LIBERO-100, mass/friction mismatch, perturbation benchmarks,
mixed-effects statistics, and the 8-task main experiment.

## Next stage

1. **Real Jev episodes** on the three bundled tasks, several init states and
   seeds, at a controlled budget — the only way to get a real answer.
2. **Horizon ablation**: sweep 0 / 0.4 / 0.8 / 1.2 / 2.0 s for success rate vs
   horizon and compute cost vs horizon. The interface already supports it.
3. **Future-information ablation**: drop checkpoints, drop events, keep only the
   final state, to find which part of the trajectory carries the signal.
4. **Expand to 8 LIBERO tasks** once the three-task mechanism is stable.
5. **Perturbation / world-model mismatch**: mass and friction offsets between
   the rollout model and the live environment, to test whether reasoning over an
   imperfect world model still helps.
