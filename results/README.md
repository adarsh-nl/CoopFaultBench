# CoopFaultBench results (v0.1.0)

Results in this directory are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

This directory holds the result payloads behind the paper's tables and
figures. It also holds the tables and figures, rebuilt from those payloads by
CPU-only scripts. Nothing here needs raw data, model weights or a GPU.

```
results/
  payloads/   result payloads (inputs to every table and figure)
  tables/     camera-ready tables (.tex/.md) and CSV exports
  figures/    figures (PDF), regenerated from the payloads
  MANIFEST.md one row per file: sha256, provenance, transformation, paper item
  SHA256SUMS  sha256sum -c compatible checksums for everything in results/
```

## Verify, then rebuild

First verify the shipped files and run the gates. The gates rebuild every
table and figure into temporary directories and compare:

```
(cd results && sha256sum -c SHA256SUMS)
python -m pytest tests/test_rebuild_tables.py tests/test_rebuild_figures.py tests/test_results_integrity.py
```

To rebuild by hand, write into a scratch directory and compare:

```
python scripts/make_tables.py --out /tmp/cfb_tables
python scripts/make_figures.py --out /tmp/cfb_figures
diff -r /tmp/cfb_tables results/tables     # differs only in model_reproduction.tex, see below
diff -r /tmp/cfb_figures results/figures   # identical with matplotlib 3.11.1
```

Both scripts write into `results/tables` and `results/figures` when `--out`
is omitted. Writing into `results/tables` replaces the camera-ready
`model_reproduction.tex` with its regeneration, which differs in two cells
(see "Known mismatch"). After that, `sha256sum -c` and the table gate report
that file until it is restored, for example with `git checkout`.

The scripts need numpy, matplotlib and the Python standard library. The
shipped PDFs were written with matplotlib 3.11.1, and with that version (and
fontTools 4.63.0) the regenerated PDFs have identical content: every object,
decompressed stream and image pixel (`tests/test_rebuild_figures.py`). Their
compressed bytes can differ across CPU architectures (Pillow's bundled zlib-ng
compresses image streams differently on ARM, e.g. Apple Silicon, than on
x86-64), so a regenerated figure's checksum can differ even when its content
is identical; on x86-64 the regenerated PDFs are also byte-identical. Other
matplotlib versions draw the same data but produce different PDF content.

`tests/test_rebuild_tables.py` is the table gate. It regenerates every table
into a temporary directory and requires byte equality with `tables/`. It also
asserts the exact set of 13 files, runs negative controls (a perturbed
payload must change the regenerated table) and traces the printed numbers
back to payload fields. One known mismatch is asserted exactly; see "Known
mismatch" below.

## Payloads

The key structure of every payload is documented in `docs/results_schema.md`.

### `payloads/fault_sweep_fixed_reference.json`: five-family fault sweep

This file is a byte-identical copy of the frozen payload (sha256 in
`MANIFEST.md`).

- **What was measured.** CoBEVT, V2X-ViT and Where2comm on the V2XSet test
  split (19 scenarios), under 16 conditions: clean, plus pose error,
  communication latency, points reduce, LiDAR fog and LiDAR snow at mild,
  moderate and severe. Each condition is scored at IoU 0.5 and 0.7.
- **Two arms.** "Individual" is ego-only (one agent). "CP" is the full
  communication-range roster. Gain = AP_CP - AP_indiv.
- **Reference set.** Every AP is scored against the fixed reference set,
  whose ground-truth objects do not depend on which agents are present.
- **Estimator.** Scenario bootstrap with B = 1000 replicates (`boot_M`) and
  seed 20260826 (`numpy.random.SeedSequence(20260826).spawn(1000)`). Each
  replicate resamples the 19 scenarios with replacement. Every `*_ci` triple
  is `[mean over replicates, 2.5th percentile, 97.5th percentile]`.
- **The tables print bootstrap means** (index 0 of each triple), not
  full-sample point estimates. The bracketed Gain interval is the 2.5-97.5
  percentile interval of the same replicates.
- **Significance.** `uncorrected_reject` means the Gain interval excludes
  zero. `holm_reject` additionally requires rejection by Holm-Bonferroni
  (alpha 0.05) over the family of 16 conditions, clean included. The
  correction is applied separately for each model and IoU, using a two-sided
  bootstrap p-value proxy: twice the smaller replicate mass at or beyond
  zero, capped at 1. In the tables, `*` marks uncorrected and `**` marks
  Holm.
- **Per-scene verdicts** (the per-scene block; its key is `KEY_PER_SCENE`
  in `coopfaultbench/reporting/payloads.py`). Frames are bootstrapped within
  each scene (B = 1000, same seed). The verdict is "established beneficial" or
  "established harmful" when that scene's Gain interval excludes zero with a
  positive or negative mean, and "not established" otherwise. A per-scene
  `holm_reject` (Holm over the 19 scenes) is stored alongside. `triggers`
  lists every "established harmful" scene verdict (209 entries).
  `halted_cells` is empty: all cells are complete.
- **AP.** Arrival-order VOC AP: OpenCOOD `calculate_ap` with
  `global_sort_detections=False`, i.e. no global sorting of detections.

### `payloads/agent_drop_fixed_vs_union.json`: agent drop, fixed vs. union reference

This file is a byte-identical copy of the frozen payload.

- **Design.** M = 100 Monte Carlo agent-drop draws with seed 20260826. In
  each draw, every non-ego in-range agent of every frame is dropped
  independently with probability p_drop. The tiers are none 0.0, mild 0.25,
  moderate 0.50 and severe 0.75. The draws use the full V2XSet test split,
  2,834 frames per model, for CoBEVT, V2X-ViT and Where2comm.
- **Scoring.** Each draw is scored twice: against the union reference (the
  standard OpenCOOD ground truth, the union of boxes over the agents
  present) and against the fixed reference. dAP = AP_fixed - AP_union is
  computed per draw.
- **Payload fields.** `mean` and `std` are the mean and standard deviation of
  dAP over the 100 draws. `ap_union.mean` and `ap_fixed.mean` are the mean
  APs.
- **The `ci` fields are not confidence intervals.** Despite the key name,
  every `ci` pair is the 2.5-97.5 percentile range of the 100 draws: a Monte
  Carlo agent-draw range. It describes how AP varies with which agents
  happen to be dropped, not uncertainty in an estimate. The released table
  labels it "Monte Carlo agent-draw range (not a confidence interval)".
- **Not printed.** `by_stratum` (per scene-agent-count dAP) is not printed
  in any released table, and no per-stratum intervals are reported.
- **Rounding.** The table prints the payload's dAP `mean` rounded to 4
  decimals. It can differ in the last digit from the difference of the two
  rounded AP means.
- **AP.** Arrival-order VOC AP, as above.

### `payloads/agent_drop_scene_bootstrap.json`: scene-level intervals for agent drop

This file was produced after the provenance tag by a CPU job that read only files of the tag (see
`MANIFEST.md`). The analysis plan, including the decision rule below, was written before the run; it is not
released, and `meta.prereg_sha256` identifies the version the job checked before computing anything. The
producing script (`meta.script_sha256`) is not part of this release either.

- **Why.** The ranges in `agent_drop_fixed_vs_union.json` hold the 19 test scenes fixed. The intervals here
  also cover which scenes happen to be in the sample.
- **Design.** A two-level bootstrap with B = 1000 replicates and seed 20260826.
  - Level 1 resamples scenes with replacement within each scene-agent-count stratum (N = 2/3/4/5 with
    7/7/3/2 scenes). One scene draw per replicate is shared by all models, drop rates, IoUs and by the
    `primary`, `frame_weighted`, `rel`, `ap_union`, `ap_fixed` and per-stratum estimators. `unstratified` makes
    its own scene draws and agent-drop realisations (see below).
  - Level 2 draws a fresh agent-drop realisation for every resampled scene copy, with the same seeding
    mechanism as the published Monte Carlo draws (replicate ids from 100 upward, disjoint from those draws).
- **Estimators** (each with `boot_mean` and `ci`, a 2.5-97.5 % percentile interval, numpy's default linear
  interpolation, no bias correction):
  - `primary`: pooled dAP = AP_fixed - AP_union over all frames of the resample. This is the estimator of the
    published `mean` values; `plugin_mc_mean` is that published mean, which the job reproduced exactly.
  - `frame_weighted`: the frame-weighted mean of per-stratum dAP, with its own Monte Carlo plug-in.
  - `unstratified`: a two-level bootstrap without stratification (19 scenes drawn from all 19, its own seed
    stream and replicate ids), a check on the stratified scheme.
  - `rel`: (AP_fixed - AP_union) / AP_union; `ap_union` and `ap_fixed`: the two APs.
- **Decision rule.** A cell is "established" (`primary.verdict`) if and only if its primary 95 % interval
  excludes zero. "Not established" does not mean "no effect"; it means not established by a scene-level
  interval on 19 scenes. `p_boot` (two-sided bootstrap proxy, min(1, 2 min(P(v <= 0), P(v >= 0)))) and
  `holm_reject_18` (Holm-Bonferroni over the 18 cells at alpha = 0.05, also requiring the interval to exclude
  zero) are secondary and never change a verdict.
  In this file 16 of 18 cells are established. The estimators disagree on one cell, reported here as a
  disagreement: `where2comm|severe|0.7` is established by `primary` [-0.0229, -0.0001] but not by
  `frame_weighted` [-0.0234, +0.0024] or `unstratified` [-0.0290, +0.0021], and Holm does not reject it.
- **Strata with few scenes.** Per-stratum intervals are computed for all four strata. For N = 4 (3 scenes) and
  N = 5 (2 scenes) they are labelled `"not interpretable: too few scenes"` (`ci_status`, `sound: false`): report
  the point value and the scene count, not the interval. These strata stay in the pooled resample, and
  resampling only 3 or 2 scenes under-disperses, so the pooled intervals somewhat understate scene-level
  uncertainty.
- **Not resampled.** Detections, matches and ground truth are fixed to the cached per-frame rows, so inference
  noise is not resampled. The 19 scenes are the whole test split: the intervals describe variability over
  scenes like these, not over other datasets.
- **`meta`.** `prereg_sha256` is the sha256 of the authors' analysis plan as the job checked it before running
  (the plan is not released; the hash lets the authors show it unchanged on request). `rows_list_sha256` and `n_row_files` identify the per-frame row files read (not
  released). `strata` lists the scenes and frame counts per stratum.

### `payloads/agent_drop_scene_bootstrap_replicates.npz`: every bootstrap replicate

The 1000 replicate values behind every interval of `agent_drop_scene_bootstrap.json`, one float64 array per
cell and estimator, named `<model>|<tier>|<iou>|<estimator>`. The estimators are `primary`, `ap_union`,
`ap_fixed`, `rel`, `frame_weighted`, `unstratified`, and per stratum `stratum_<N>`, `stratum_<N>_ap_union`,
`stratum_<N>_ap_fixed` and `stratum_<N>_rel`. The tier `none` (p_drop = 0) is included; the job checked that
every pooled dAP replicate is exactly 0 there. `tests/test_results_integrity.py` checks that each primary
interval is the 2.5 and 97.5 percentile of its array. Load it with `numpy.load` (no pickled objects); the array
names contain `|`, so extracting the archive with an unzip tool fails on Windows.

### `payloads/reference_shrinkage.json`: how much the union reference shrinks under agent drop

This file was produced after the provenance tag by a CPU job that read only files of the tag (see
`MANIFEST.md`); the producing script (`meta.script_sha256`) is not part of this release. Everything is per
model, because each model's rows come from its own data-loading run, so in-range rosters and ground-truth
counts can differ slightly (V2X-ViT and Where2comm give identical values here; CoBEVT differs on a few frames).

- **Quantity.** For a set of frames, shrinkage = sum of n_gt(union reference of the realised roster) / sum of
  n_gt(fixed reference) - 1, where the fixed reference is the union over the full in-range roster. It is at
  most 0: minus the share of ground-truth boxes that leave the standard (union) reference when agents drop
  out. Frames are pooled as a ratio of sums. Ground-truth counts do not depend on the IoU threshold (the job
  checks this on every configuration).
- **Agent drop.** At p_drop = 0.25 / 0.50 / 0.75, over the same M = 100 draws as `agent_drop_fixed_vs_union.json`.
  The job reproduced all 24 dAP means of that payload exactly from these draws, so the shrinkage and the dAP of
  a drop rate come from the same realisations. `mean`, `std` and `range_2.5_97.5` are taken over the draws,
  pooled and per stratum; the range is a Monte Carlo agent-draw range, not a confidence interval.
- **Ego only.** Every collaborator dropped. It is deterministic, pooled and per stratum.
  `frames_with_no_collaborator_in_range` counts the frames whose in-range roster is the ego alone.
- **Geometry.** `frac_frames_with_a_cav_cut` is the share of frames in which the scene has more agents than the
  ego's in-range roster (at least one agent beyond the communication range). `r_pooled` is the Pearson
  correlation over all frames between the in-range roster size (ego included) and the fixed reference's box
  count. `r_by_stratum` gives the same per stratum, for strata with at least 5 frames whose roster size varies;
  `r_within_stratum` is their frame-weighted mean.

### `payloads/fault_inference_grid.csv`: fault-inference grid (109 rows)

This file is the grid table with its `job_id` column removed. Every other
field of every row is byte-for-byte unchanged.

- **CoBEVT, V2X-ViT and Where2comm** (V2XSet test, 2,834 frames) have 31 rows
  each: clean, six injectors at three tiers, and agent drop reported per
  scene-agent-count stratum (2/3/4/5 agents) at three tiers.
- **CoSDH** (OPV2V test, 2,170 frames) has 16 rows: clean and five injectors
  at three tiers.
- **Run setup.** There is one evaluation pass per cell, with seed 1234, and
  AP@0.5/0.7 is scored against the standard (union) reference. The V2XSet
  models were evaluated without global detection sorting. The CoSDH cells
  use the CoSDH release's own evaluation path, unmodified.
- **Columns.** `faulty_ap50` and `faulty_ap70` are the per-cell `ap_50` and
  `ap_70`, or the matching stratum for agent drop, rounded to 6 decimals.
  `rel_drop_ap70 = delta_ap70 / clean_ap70`. `extra_metric` holds
  injector-specific diagnostics, such as `scene_agent_count`, `n_scenes` and
  `removed_frac`.

### `payloads/cells/<model>/<injector>/<tier>.json`: per-cell results (98 files)

These are the per-cell results behind the grid: 26 files each for CoBEVT,
V2X-ViT and Where2comm, and 20 for CoSDH.

- **Extra cells.** `none/clean_b`, `none/clean_c`, `none/clean_rep` and
  `none/wrapped_null` are determinism and no-op-wrapper control cells, which
  are not printed in any table.
- **Removed keys.** Four keys were removed: `model_dir` and `opencood_path`
  (scheduler-local and home-directory paths), `job_id`, and `frame_stratum`
  (a per-frame list).
- **Unchanged.** All other keys keep their original order and values. Only
  `none/clean.json` is read by a table: its `wild_setting_shipped` gives the
  evaluation-setting columns of `model_reproduction.tex`.

### `payloads/fault_inference_models.json`: per-model fields of the grid manifest

This file holds `dataset`, `setting`, `expected_frames`, `graded` and
`published` for each model, plus the list of not-applicable cells. The rest
of the grid manifest (hostnames, job ids, file-system paths, internal notes)
is not released. `model_reproduction.tex` reads it.

## CoSDH caveat

- **Checkpoint.** The CoSDH checkpoint is the one published by the CoSDH
  authors. Their README says verbatim:
  "These checkpoints have been retrained and the results differ slightly from those reported in the paper."
- **Grading.** CoSDH is therefore listed as not graded against the published
  numbers.
- **Missing cells.** CoSDH has no agent-drop or missing-modality cells: all
  six failed in the original runs, in the released CoSDH inference path. The
  cells are absent from the grid, the per-cell results and every table and
  figure, and are never filled. They print as `N/A` in
  `severe_tier_ordering.tex`, and CoSDH is absent from
  `agent_drop_by_stratum.*` and from the missing-LiDAR curve of
  `degradation_all_models.pdf`.
- **Comparability.** CoSDH is evaluated on OPV2V, while the other three
  models use V2XSet. Relative drops normalise baselines, not datasets.

## Tables (`tables/`)

| File | Content | Source |
|---|---|---|
| `fault_sweep_cobevt.tex`, `fault_sweep_v2xvit.tex`, `fault_sweep_where2comm.tex`, `fault_sweep_tables.md` | Per-model fault-sweep AP/Gain tables (bootstrap means; 95% percentile interval; Holm over 16 conditions) | camera-ready copies (codenames renamed, see below); regenerate byte-identically |
| `fault_sweep_<model>.csv` | The same cells at full precision | generated |
| `agent_drop_fixed_vs_union.{tex,md,csv}` | 18 rows: 3 models x p_drop 0.25/0.50/0.75 x IoU 0.5/0.7; AP_union, AP_fixed, dAP (means over 100 draws, 4 dp) with Monte Carlo agent-draw ranges | generated (new) |
| `model_reproduction.tex` | Models, shipped evaluation settings, clean AP vs. released benchmark AP, grid completion | camera-ready copy; see Known mismatch |
| `severe_tier_ordering.tex` | Severe-tier relative AP@0.7 drop by injector and model, incl. CoSDH | camera-ready copy; regenerates byte-identically |
| `agent_drop_by_stratum.tex` | Agent drop by scene agent count, union reference | camera-ready copy; regenerates byte-identically |

Reading notes for two camera-ready tables (printed as in the paper; nothing changed):

- `model_reproduction.tex`: the "Graded" column says "yes" for Where2comm, because the table grades it against
  OpenCOOD's released V2XSet benchmark, while the model record in `payloads/fault_inference_models.json` has
  `"graded": false` (its V2XSet checkpoint is a third-party training, see `docs/model_provenance.md`). The
  "Cells" column is the model's number of rows in `payloads/fault_inference_grid.csv` over the 26 cells attempted
  for it. The 31 rows are 1 clean row, 3 tiers for each of the six other injectors, and agent drop reported per
  tier and agent-count stratum (3 x 4 rows), so the ratio can exceed 1.
- `severe_tier_ordering.tex`: the "Mean" column averages four models for the first five injectors but three for
  "Missing LiDAR" and "Agent drop", where CoSDH has no cells; the ranking therefore compares means over
  different model sets.

The camera-ready copies differ from the original files only where those
files named internal project codenames or private-repository files: one
LaTeX label per fault-sweep table, the markdown headings, and the
generator comment on the first line. These were renamed to plain names. The
exact replacement list is kept in the authors' records; `MANIFEST.md` gives
the sha256 of the original and of the shipped file. No number, caption
sentence, row or column is changed.

## Figures (`figures/`)

| File | Content |
|---|---|
| `fault_sweep_gain_forest.pdf` | Per-condition individual vs. CP AP@0.7 dumbbells. The connector is solid when the Gain is Holm-significant |
| `fault_sweep_scene_verdict.pdf` | Per-scene verdict grid at AP@0.7 |
| `fault_sweep_degradation.pdf` | AP@0.7 vs. severity, individual vs. CP, with bootstrap percentile bands |
| `fault_sweep_scene_verdict_composition.pdf` | Fraction of the 19 scenes in each verdict, at IoU 0.5 and 0.7 |
| `agent_drop_fixed_vs_union.pdf` | AP@0.7 vs. agent-drop severity, fixed vs. union reference (means over 100 draws) |
| `agent_drop_by_stratum.pdf` | Agent drop by scene agent count (union reference) |
| `severe_tier_ordering.pdf` | Severe-tier relative AP@0.7 drop by injector and model |
| `degradation_all_models.pdf` | AP@0.7 vs. severity per injector, all four models (supplementary) |

With matplotlib 3.11.1, seven of the eight are byte-identical to the
camera-ready PDFs (`MANIFEST.md` records both hashes). The eighth,
`fault_sweep_scene_verdict_composition.pdf`, differs only in its title, which
was reworded to "Per-scene verdict composition" because the camera-ready
title used an internal shorthand. Rendered with the camera-ready title, it
is byte-identical as well.

## Known mismatch

`model_reproduction.tex` is the camera-ready table. It prints `n/a` for
Where2comm's published AP@0.5/@0.7. The grid manifest in the frozen
provenance checkout records `published = {ap50: 0.82, ap70: 0.534}` for
Where2comm: the OpenCOOD README V2XSet benchmark row, against which the table
grades Where2comm. Regenerating from the released payloads therefore prints
`0.8200 & 0.5340` in those two cells. Every other cell matches. The numbers
were not edited to force agreement: the shipped file is the camera-ready
table, and `tests/test_rebuild_tables.py` asserts that these two cells are
the only difference.

## Not released here

- **Excluded results.** Results that need the 400-frame stratified
  ground-truth study, raw point clouds or boxes, per-frame rows, model
  weights or a GPU are not released here.
- **Excluded figure variants.** Alternative, unapproved design variants of
  the agent-drop stratification figure are not released.
- **Excluded tables.** The remaining grid tables of the paper (injector
  parameters, wrapped-null gate, determinism floors, pose asymmetry, fog
  defect, snow mechanism, scene counts) are not part of this result set.
