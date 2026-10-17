# Methodology

## Union vs fixed reference

For a frame with full roster A (the agents within the 70 m communication range of the ego) and an evaluated
subset A' (for example after agent drop):

- the **union reference** G(A') is the set of ground-truth objects contributed by the agents in A'. This is
  what OpenCOOD's evaluation uses: the reference changes with the agent set.
- the **fixed reference** G(A) is the union over the full roster, held constant across every subset of the
  frame.

The same detections are scored against both. dAP = AP_fixed − AP_union. Because G(A') ⊆ G(A), dropping
agents shrinks the union reference and typically makes AP_union look better than AP_fixed.

## Average precision

VOC-2010 AP (`coopfaultbench.evaluation.voc_ap`) over match flags concatenated in **arrival order** — the
order in which frames were scored — without a global re-sort by score (OpenCOOD's
`global_sort_detections=False`). AP therefore depends on the frame order; all results use the dataset's own
order. Matching against a reference uses OpenCOOD's matcher (`caluclate_tp_fp`) in the OpenCOOD drivers.

## Agent drop under both references (`results/payloads/agent_drop_fixed_vs_union.json`)

- `scripts/run_agent_subset_sweep.py` evaluates, for every frame, **every** agent subset containing the ego and
  scores each realised subset against both references. The realised roster is read from OpenCOOD's own data
  path (after its communication-range filter), not recomputed.
- `scripts/resample_agent_drop.py` then draws i.i.d. agent drop (p = 0.25 / 0.50 / 0.75, M = 100 replicates;
  each keep/drop decision seeded by blake2s of `frame|agent|replicate|tier`) and looks up the cached
  configuration of the realised roster. Because the communication-range test is per agent, every draw has an
  exact cached match; a missing one is an error.
- Per model, drop rate and IoU the payload gives the mean, std and 2.5–97.5 % range over the M replicates of
  dAP, AP_union and AP_fixed, and per scene-agent-count stratum the mean and std of dAP. The range reflects
  which agents are dropped with the scenes held fixed: it is **not** a confidence interval over scenes.
- Two payloads produced after the provenance tag complete this analysis (`results/README.md`):
  - `agent_drop_scene_bootstrap.json`: scene-level 95 % intervals from a two-level bootstrap over the 19
    test scenes (stratified by agent count; a fresh agent-drop draw per resampled scene), with every
    replicate in `agent_drop_scene_bootstrap_replicates.npz`.
  - `reference_shrinkage.json`: how much the union reference loses under the same agent-drop draws, and
    with every collaborator dropped.

## Five-family fault sweep (`results/payloads/fault_sweep_fixed_reference.json`)

Three V2XSet models × 16 conditions (clean + 5 families × 3 tiers) × IoU 0.5 / 0.7, all scored against the
**fixed** reference, on the full test split (19 scenarios, 2,834 frames). For each condition the payload holds
the AP with only the ego's data and with the full roster, their difference (the cooperative gain), scene-level
bootstrap means and 95 % intervals (B = 1000, seed 20260826, scenes resampled), and Holm correction across the
16 conditions per model and IoU (the clean condition included). **The tables print the bootstrap means**, which
differ slightly from the plug-in values. Per-scene verdicts use the uncorrected intervals. The payload contains
fixed-reference values only; union-reference values were not computed for this sweep.

## Determinism

Injector draws are per (frame, agent, stage). OpenCOOD's own point shuffling draws from numpy's global random
state; the inference driver seeds it once per run, the agent-subset sweep per frame and agent. GPU inference is not bit-reproducible between runs (up to
about 0.03 m in box corners); regression checks therefore run on cached match flags, not on inference.
