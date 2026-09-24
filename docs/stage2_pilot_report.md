# Stage 2 Pilot — Representation, Future Information, and Correspondence

Stage 1 asked how far ahead to reason. Stage 1.5 found that how the future is
*built* and *written down* were confounded with the question. Stage 2 separates
the three candidate explanations with one paired design:

```
S1Original
    ↓  only the representation changes
S1Compact
    ↓  only a correct multi-step future is added
PolicyCompact
    ↓  only the action↔future correspondence is destroyed
PolicyShuffle
```

Pilot scale: 3 tasks × 3 fixed initial states × 4 arms = **36 paired episodes**,
plus a separate 9-episode repeatability probe.

---

## 1. Execution integrity

| | |
|---|---|
| Commit at run time | **`2cf4b97`** (working tree clean) |
| Stage 2 implementation | `7f1fb1e` |
| Bug fix applied before any data | `2cf4b97` (see below) |
| Offline tests | **95 passed**, 49 skipped |
| Simulation tests | 139 passed, 8 failed — all attributable, see below |
| Provider | TypeSafe |
| Requested model | `jev-latest` (rolling alias) |
| Tasks | `top_drawer`, `microwave`, `alphabet_soup` |
| Initial states | 0, 1, 2 (LIBERO fixed states; 50 available per task) |
| Decision cap | 60, identical for all four arms |
| Arm order | deterministic per-block permutation, seed 20260923 |

### A bug was found and fixed before the pilot produced any data

The post-hoc request signatures caught it. `CompactFormattingAPI` renames
`future_trajectory` to `future`, and in the runner's wrapper stack it runs
*before* `ShufflingTrajectoryAPI`. The shuffle matched only the old key, found
no carriers, and returned the criteria untouched — so **`policy_shuffle` was
sending exactly the assignment `policy_compact` sent** while being reported as
its negative control. Comparison C would have been meaningless and nothing
would have failed.

The run was stopped at **zero completed episodes**, the shuffle was fixed to
match whichever future key is present, three regression tests were added
(the earlier tests exercised the shuffle in isolation, which is why a
composition bug survived them), and the pilot restarted from the beginning.
Committed separately as `2cf4b97`.

### Test failures are not regressions

Seven of the eight simulation failures are the known upstream replay drift. The
drift on this host is now `max_state_error = 8.17065030273012e-07`, five orders
of magnitude larger than the 6.44e-12 recorded earlier in the project — the
machine's floating-point behaviour changed, probably at a reboot. Re-running the
same tests on the **unmodified upstream checkout `3bdad98`** produces the
identical value, so it is not a WorldRobot regression.

The eighth, `test_isolated_noninteractive_configuration`, **passes when run
alone**; it fails only in a full session, from LIBERO's global configuration
state being set by an earlier test. Pre-existing test interaction, not Stage 2.

### Pre-registered variables were the only ones that changed

Signatures computed after the fact from the saved request logs, never injected
into the run. Comparisons are restricted to the **aligned prefix** — the
decisions before two arms first choose differently, after which they are in
different states and any divergence is expected rather than informative.

| block | aligned A / B / C | pool equal | shared immediate equal | only policy has future | future set equal | assignment differs |
|---|---|:---:|:---:|:---:|:---:|:---:|
| alphabet_soup init 0 | 25 / 25 / 19 | ✅ | ✅ | ✅ | ✅ | ✅ |
| alphabet_soup init 1 | 13 / 13 / 13 | ✅ | ✅ | ✅ | ✅ | ✅ |
| alphabet_soup init 2 | 26 / 18 / 19 | ✅ | ✅ | ✅ | ✅ | ✅ |
| microwave init 0 | 9 / 12 / 4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| microwave init 1 | 26 / 6 / 6 | ✅ | ✅ | ✅ | ✅ | ✅ |
| microwave init 2 | 4 / 7 / 4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| top_drawer init 0 | 32 / 32 / 5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| top_drawer init 1 | 7 / 7 / 5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| top_drawer init 2 | 8 / 8 / 5 | ✅ | ✅ | ✅ | ✅ | ✅ |

**9 blocks, 0 violated invariants.** Oracle leakage in state and criteria is
zero for every arm.

Two earlier versions of this check reported violations, and both were the
checker's fault rather than the experiment's:

1. It compared all decisions rather than the aligned prefix. Past the first
   disagreement the arms are in different states, so different menus are
   correct.
2. It compared the *motor menu*, which the pipeline filters by the chosen
   motion-family strategy. Two arms in the same state with different strategy
   choices are legitimately offered different motor menus while the underlying
   feasible pool is identical. Fairness is a property of the pool; the check now
   compares pools, and compares the immediate facts only over candidates both
   arms were offered.

Strategy-layer divergences while states were aligned, recorded as a result
rather than a fault: 4 in total across 9 blocks (microwave init 0: B=1;
microwave init 2: A=1, C=1; top_drawer init 1: A=1, B=1).

### Reproducibility limitation

> The provider exposes only a rolling `jev-latest` alias, and does not return
> the resolved underlying model version. The run is reproducible in
> configuration but not in model version.

Recorded in `results/stage2/run_metadata.json` with `model_version_pinned: false`.

---

## 2. Raw results

Every episode, not just averages. `dec2s` is the decision at which the LIBERO
predicate first became true; `—` means the episode never succeeded and is
right-censored.

| block | arm | success | dec2s | censored | tok/dec | cost | termination |
|---|---|:---:|---:|:---:|---:|---:|---|
| alphabet_soup init 0 | s1_original | ✅ | 48 | | 2,060 | $0.00415 | success |
| | s1_compact | ✅ | 44 | | 1,956 | $0.00361 | success |
| | **policy_compact** | ✅ | **41** | | 3,480 | $0.00599 | success |
| | policy_shuffle | ✅ | 58 | | 2,926 | $0.00712 | success |
| alphabet_soup init 1 | s1_original | ❌ | — | ✔ | 1,954 | $0.00106 | no_feasible_action |
| | s1_compact | ❌ | — | ✔ | 1,954 | $0.00106 | no_feasible_action |
| | policy_compact | ❌ | — | ✔ | 3,182 | $0.00173 | no_feasible_action |
| | policy_shuffle | ❌ | — | ✔ | 3,182 | $0.00173 | no_feasible_action |
| alphabet_soup init 2 | s1_original | ❌ | — | ✔ | 1,765 | $0.00192 | no_feasible_action |
| | s1_compact | ❌ | — | ✔ | 1,765 | $0.00192 | no_feasible_action |
| | policy_compact | ❌ | — | ✔ | 3,111 | $0.00535 | no_feasible_action |
| | policy_shuffle | ❌ | — | ✔ | 2,939 | $0.00481 | no_feasible_action |
| microwave init 0 | s1_original | ✅ | 14 | | 1,734 | $0.00101 | success |
| | s1_compact | ✅ | 14 | | 1,927 | $0.00113 | success |
| | policy_compact | ✅ | 14 | | 3,354 | $0.00197 | success |
| | policy_shuffle | ✅ | 19 | | 3,095 | $0.00246 | success |
| microwave init 1 | s1_original | ✅ | 29 | | 1,586 | $0.00193 | success |
| | s1_compact | ✅ | 40 | | 1,528 | $0.00256 | success |
| | **policy_compact** | ✅ | **13** | | 3,582 | $0.00195 | success |
| | policy_shuffle | ✅ | 27 | | 3,394 | $0.00384 | success |
| microwave init 2 | **s1_original** | ✅ | **44** | | 1,697 | $0.00313 | success |
| | s1_compact | ❌ | — | ✔ | 1,871 | $0.00408 | no_feasible_action |
| | policy_compact | ❌ | — | ✔ | 2,566 | $0.00646 | decision_limit |
| | policy_shuffle | ❌ | — | ✔ | 3,932 | $0.00990 | decision_limit |
| top_drawer init 0 | s1_original | ✅ | 32 | | 1,506 | $0.00202 | success |
| | s1_compact | ✅ | 32 | | 1,470 | $0.00197 | success |
| | policy_compact | ✅ | 32 | | 2,482 | $0.00333 | success |
| | policy_shuffle | ✅ | 33 | | 3,188 | $0.00441 | success |
| top_drawer init 1 | s1_original | ✅ | 33 | | 1,751 | $0.00242 | success |
| | **s1_compact** | ✅ | **24** | | 1,430 | $0.00144 | success |
| | policy_compact | ❌ | — | ✔ | 2,907 | $0.00354 | no_feasible_action |
| | policy_shuffle | ✅ | 25 | | 4,102 | $0.00430 | success |
| top_drawer init 2 | s1_original | ✅ | 45 | | 1,668 | $0.00315 | success |
| | s1_compact | ❌ | — | ✔ | 1,775 | $0.00447 | decision_limit |
| | policy_compact | ✅ | 43 | | 2,321 | $0.00419 | success |
| | policy_shuffle | ❌ | — | ✔ | 2,502 | $0.00630 | decision_limit |

### Per method

| arm | success | mean dec2s (solved only) | tok/dec | cost |
|---|---|---:|---:|---:|
| **s1_original** | **7/9** | 37.89 | 1,747 | $0.0208 |
| s1_compact | 5/9 | 33.00 | 1,742 | $0.0223 |
| policy_compact | 5/9 | **30.67** | 2,998 | $0.0346 |
| policy_shuffle | 5/9 | 36.67 | 3,251 | $0.0449 |

### Completion curve

Cumulative successes out of 9, at each shared decision budget.

| arm | @10 | @20 | @30 | @40 | @50 | @60 |
|---|---:|---:|---:|---:|---:|---:|
| s1_original | 0 | 1 | 2 | 4 | **7** | **7** |
| s1_compact | 0 | 1 | 2 | 4 | 5 | 5 |
| policy_compact | 0 | **2** | 2 | 3 | 5 | 5 |
| policy_shuffle | 0 | 1 | **3** | 4 | 4 | 5 |

Two `alphabet_soup` blocks (init 1 and 2) failed identically in all four arms on
`no_feasible_action`, so they carry no comparative information and reduce the
effective sample to seven blocks.

## 3. Comparison A — Representation Effect

**S1Original vs S1Compact.** Same facts, different serialization: field names
move to a legend sent once in the state instead of being repeated on every
candidate. `expand(compact(x)) == x` is asserted on real recorded prompts, and
the signature check confirms shared candidates carried identical facts.

| | |
|---|---|
| paired blocks | 9 |
| successes | s1_original **7** vs s1_compact **5** |
| blocks both solved | 5 |
| compact faster / slower / tied | 2 / 1 / 2 |
| mean decision delta | **−0.4** (compact faster) |
| bootstrap CI (descriptive) | [−5.4, +5.8] |
| tokens per decision | 1,747 vs 1,742 |

**The compaction did not deliver the token saving it was built for.** 1,742
against 1,747 is a 0.3% difference, not the reduction the legend design
predicted. The measured saving on recorded prompts was real but small, and it is
swamped by the rest of the request — the state block, the instructions, and the
intent/strategy layers, none of which the compaction touches.

**On the primary outcome it looks worse, and the two losses are informative.**
s1_compact failed two blocks that s1_original solved (microwave init 2,
top_drawer init 2), and won one (top_drawer init 1, 24 vs 33). Among blocks both
solved, the decision counts are essentially tied and the interval spans zero in
both directions.

So representation did change behaviour — the arms diverge, sometimes early — but
this pilot gives **no evidence that lossless compaction helps**, and a weak
signal that it may cost success. With n=7 effective blocks and a 2-episode
success gap, that reading is not secure.

## 4. Comparison B — Marginal Multi-step Future Value

**S1Compact vs PolicyCompact.** This is the main result. Both arms share the
compact immediate representation, the candidate set, the one-step information
and the task state; only PolicyCompact carries a policy-consistent multi-step
future. The signature check confirms it at the request level: shared candidates
carry identical immediate facts, and only PolicyCompact's options contain a
`future` key.

| | |
|---|---|
| paired blocks | 9 |
| successes | s1_compact **5** vs policy_compact **5** |
| blocks both solved | 4 |
| policy faster / slower / tied | **2 / 0 / 2** |
| mean decision delta | **−7.5** (policy faster) |
| bootstrap CI (descriptive) | [−20.25, 0.0] |
| tokens per decision | 1,742 vs 2,998 (**1.7×**) |

**Where both solved, adding the future never cost decisions and sometimes saved
many.** The two wins are microwave init 1 (13 vs 40, a 27-decision saving) and
alphabet_soup init 0 (41 vs 44). The other two blocks tied exactly. The interval
touches zero at its upper end, so "no effect" is not excluded.

**Success counts are level at 5/9, and the two arms fail different blocks.**
s1_compact failed microwave init 2 and top_drawer init 2; policy_compact failed
microwave init 2 and top_drawer init 1. They are not strictly ordered.

The honest reading: **among blocks both arms solved, the correct multi-step
future consistently did not hurt and twice helped substantially — one of those
being a 3× reduction. It did not convert any failure into a success, and it
costs 1.7× the tokens.**

## 5. Comparison C — Action–Future Correspondence

**PolicyCompact vs PolicyShuffle.** Identical candidate sets, identical one-step
information, identical generated future sets, comparable token volume. The only
change is which future sits beside which candidate, by deterministic
derangement. The signature check confirms the future *sets* are identical while
the *assignments* differ, at every decision where at least two candidates
carried a future.

| | |
|---|---|
| paired blocks | 9 |
| successes | policy_compact **5** vs policy_shuffle **5** |
| blocks both solved | 4 |
| shuffle faster / slower / tied | **0 / 4 / 0** |
| mean decision delta | **+9.25** (shuffle slower) |
| bootstrap CI (descriptive) | [+3.0, +15.5] |
| tokens per decision | 2,998 vs 3,251 |

**Every block that both arms solved was slower with the correspondence broken,
with no exceptions and no ties.** The interval excludes zero on the descriptive
bootstrap. This is the cleanest directional signal in the pilot.

It is also the comparison with the least room for an alternative explanation:
the physics, the menu and the prompt shape are held identical by construction
and verified by hash, so the difference is attributable to *which* future was
shown beside *which* action.

**The caveat that keeps this from being conclusive**: success counts are again
level at 5/9, and the shuffle arm actually solved one block policy_compact
failed (top_drawer init 1). Four paired observations, all pointing the same way,
is a direction — not an effect size anyone should quote.

<!--PLACEHOLDER:repeat-->

## 7. Future novelty — exploratory only

`future_novelty_count` remains diagnostic. It is never sent to Jev, never used
to rank candidates, never used to gate or change a rollout.

Across the 128 decisions where S1Compact and PolicyCompact were in aligned
states, they chose the same action **120 times** and differed **8 times**.

| | decisions | mean future_novelty_count |
|---|---:|---:|
| the two arms agreed | 120 | 0.93 |
| the two arms differed | 8 | **1.50** |

The eight disagreements carried more novel signal in the chosen candidate's
future than the agreements did. That is the direction one would expect if the
trajectory is what moved the choice — but eight observations is far too few to
lean on, and no causal claim is made. It is recorded to show the diagnostic
behaves sensibly, not as evidence.

**The more striking number is 120/128 agreement.** For the great majority of
aligned decisions, adding a correct multi-step future did not change what
PolicyCompact chose. Whatever value the future carries in Comparison B is
concentrated in a small minority of decisions.

<!--PLACEHOLDER:branch-->

<!--PLACEHOLDER:files-->
