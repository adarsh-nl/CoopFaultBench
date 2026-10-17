# CoopFaultBench

Fault injection and **dual-reference (fixed vs union) evaluation** for cooperative (multi-agent) 3D
perception, together with the results and table/figure sources of the WACV 2027 paper

> **Agent-Dependent Ground Truth Biases Robustness Evaluation in Collaborative Perception**
> Adarsh Nanjaiya Latha, Alessandro Chiumento, Jan Laarhuis, Ozlem Durmaz Incel. WACV 2027.

Cooperative perception is evaluated against a ground-truth set that the standard toolchain assembles as the
union of what the participating agents see. When a fault or an ablation removes an agent, that reference
shrinks with it, so recall is inflated and losing a collaborator looks less damaging than it is.
CoopFaultBench scores the same detections against the usual **union reference** and against a **fixed
reference** (the union over the frame's full roster) and reports both.

What is in this repository:

| | |
|---|---|
| `coopfaultbench/injectors/` | fault injectors: pose error, communication latency, agent drop, missing modality, LiDAR point reduction, LiDAR fog and snow, camera corruptions, sensor occlusion, calibration error, temporal misalignment |
| `coopfaultbench/faults.py`, `configs/fault_grid.yaml` | the fault conditions of the paper and their reproducible per-(frame, agent) seeding |
| `coopfaultbench/evaluation/` | dual-reference scoring, arrival-order VOC AP, Monte Carlo agent drop |
| `coopfaultbench/opencood/`, `scripts/run_*.py` | running the faults inside [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) (installed separately) |
| `results/` | the paper's result payloads, tables and figures (CC BY 4.0), with a SHA-256 manifest |
| `scripts/make_tables.py`, `scripts/make_figures.py` | rebuild every table and figure in `results/` (the paper's camera-ready result set, see `results/README.md`) on a CPU |

No dataset content and no model weights are distributed.

## Install

Python 3.7 or newer (on Python 3.7 the editable install needs pip >= 21.3). The core needs numpy, scipy, PyYAML
and Pillow.

```bash
git clone https://github.com/adarsh-nl/CoopFaultBench.git
cd CoopFaultBench
pip install -e ".[test,figures]"          # add ,camera for the camera corruptions
pytest                                   # about a minute on a laptop CPU
```

For a byte-for-byte reproduction of the tables, and figures with identical content (their compressed bytes can
differ across CPU architectures; `docs/limitations.md`), use the pinned environment the results were produced
with (Python 3.13):

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
```

## Five-minute quickstart (CPU, no data needed)

```python
import numpy as np
from coopfaultbench.injectors import PointsReductionInjector, LidarFogInjector, PoseErrorInjector
from coopfaultbench.sample import CooperativeSample, AgentFrame
from coopfaultbench.faults import FaultSpec, apply_sample_faults
from coopfaultbench.evaluation import union_reference, fixed_reference, score_against_references, dual_ap

# 1. Injectors work on plain arrays: an (N, 4) point cloud [x, y, z, intensity] ...
rng = np.random.default_rng(0)
points = np.c_[rng.uniform(-40, 40, (5000, 2)), rng.uniform(-2, 1, 5000), rng.uniform(0, 1, 5000)]
kept = PointsReductionInjector(severity=2, seed=1)(points)          # keeps 20 % of the points
foggy = LidarFogInjector(severity=3, seed=1)(points)                 # non-commercial code path, see Licence

# 2. ... or on a cooperative sample: the paper's sample-stage fault conditions, seeded per (frame, agent,
#    stage). (Latency selects which frame is read, so it acts in the data loader, see coopfaultbench.opencood.)
sample = CooperativeSample(frame_index=0, ego_id='ego')
for aid, x in (('ego', 0.0), ('cav1', 20.0)):
    pose = np.eye(4); pose[0, 3] = x
    sample.agents[aid] = AgentFrame(agent_id=aid, is_ego=(aid == 'ego'), pose=pose, lidar=points.copy())
apply_sample_faults(sample, FaultSpec(seed=1234, pose_error={'sigma_xy': 0.4, 'sigma_heading': 0.4}), 0)
print(sample.agents['cav1'].faults)                                  # the collaborator moved, the ego did not

# 3. Dual-reference scoring. Objects A, B, D are visible to the ego, C only to the collaborator.
visible = {'ego': {'A', 'B', 'D'}, 'cav1': {'C', 'D'}}
union = sorted(union_reference(visible, ['ego']))                    # collaborator dropped: C leaves the reference
fixed = sorted(fixed_reference(visible, ['ego', 'cav1']))
hits = ['A', 'B', 'D']                                               # three correct detections
iou = lambda ref: np.array([[0.9 if h == o else 0.0 for o in ref] for h in hits])
s = score_against_references([0.9, 0.8, 0.7], iou(union), iou(fixed))
print(dual_ap([s]))     # AP_union = 1.0, AP_fixed = 0.75, dAP = -0.25
```

Rebuild the paper's tables and figures from `results/` (verify the shipped files first, then rebuild into a
scratch directory and compare):

```bash
(cd results && sha256sum -c SHA256SUMS)
python scripts/make_tables.py --out /tmp/cfb_tables      # every table, from results/payloads/
python scripts/make_figures.py --out /tmp/cfb_figures    # every figure (needs the `figures` extra)
diff -r /tmp/cfb_tables results/tables                   # one documented difference, see results/README.md
```

## Rebuilding the paper tables

`results/` holds the result payloads: the two main payloads (fault sweep, agent drop) byte-identical to what the
analyses wrote, and the fault-inference grid files sanitized (job ids and local paths removed). Every file is
traced by SHA-256 in `results/MANIFEST.md`. The scripts above regenerate every released table and figure from
them; `tests/test_rebuild_tables.py` checks that the regenerated tables equal the shipped ones (one documented
cell pair differs, see `results/README.md`). Read `results/README.md` before quoting a number:

- The five-family fault-sweep tables print **bootstrap means** (B = 1000 scene-level replicates, seed
  20260826) on the **fixed reference**, with Holm correction over the 16 conditions (clean included) per model
  and IoU.
- The agent-drop table gives the mean over **M = 100 Monte Carlo agent-drop draws** on the full test split
  (2,834 frames per model). Its brackets are the 2.5–97.5 % range of those draws: a Monte Carlo agent-draw
  range, **not** a confidence interval over scenes.
- AP is VOC-2010 AP over detections in **arrival order** (OpenCOOD's `global_sort_detections=False`), so it
  depends on frame order (see `docs/methodology.md`).
- CoSDH (OPV2V) uses the checkpoint its authors released, which they state was retrained; it has no
  agent-drop or missing-modality cells (`results/README.md`).

## Running a fault sweep with OpenCOOD

The faults run inside OpenCOOD's own, unmodified inference: `scripts/run_fault_inference.py` swaps OpenCOOD's
dataset class for a subclass that injects the faults (`coopfaultbench.opencood.faulty_dataset`) and then calls
OpenCOOD's `inference.main()`. You need:

1. OpenCOOD, installed from <https://github.com/DerrickXuNu/OpenCOOD> under its own licence (an academic,
   non-transferable licence; nothing of OpenCOOD is included here), with its dependencies (PyTorch, spconv);
   for V2X-ViT the code base <https://github.com/DerrickXuNu/v2x-vit> (namespace `v2xvit`, `--fi_ns v2xvit`).
   CoSDH (<https://github.com/Xu2729/CoSDH>, an OpenCOOD fork) has its own driver,
   `scripts/run_cosdh_inference.py`, run with the CoSDH checkout first on `PYTHONPATH`.
2. A trained checkpoint directory (`config.yaml` + weights): see `docs/model_provenance.md`.
3. The dataset (see below), with the checkpoint's `config.yaml` pointing at your copy of the test split.

```bash
pip install -e .                                 # into OpenCOOD's Python environment
python scripts/run_fault_inference.py --model_dir /path/to/cobevt --fusion_method intermediate \
    --fi_condition lidar_fog/severe --fi_out out/cobevt/lidar_fog/severe
python scripts/run_fault_inference.py --model_dir /path/to/cobevt --fusion_method intermediate \
    --fi_condition official --fi_out out/cobevt/official     # the unwrapped control
```

`--fi_condition` takes `official`, `clean` (the wrapper with no fault) or `<injector>/<tier>` from
`coopfaultbench/configs/fault_grid.yaml` (pose_error, latency, agent_drop, missing_modality, points_reduce,
lidar_fog, lidar_snow × mild, moderate, severe). Each run writes `fi_result.json` with AP@0.3/0.5/0.7, the
per-stratum APs and the fingerprints of the code that ran.

For dual-reference scoring under agent drop, `scripts/run_agent_subset_sweep.py` scores every agent subset of
every frame against both references, and `scripts/resample_agent_drop.py` turns those rows into AP_union,
AP_fixed and dAP per drop rate on a CPU. For the five-family fault sweep, `scripts/run_fault_sweep.py --fault
<family> --tier <tier>` scores the ego-only and full-roster arms under each condition, and
`scripts/analyze_fault_sweep.py` turns the rows of all conditions into the fault-sweep payload (scene
bootstrap, Holm correction, per-scene verdicts) on a CPU.

Which agents a fault touches: `points_reduce`, `lidar_fog` and `lidar_snow` hit **all agents including the
ego** (sensor/environment faults); `pose_error`, `latency`, `agent_drop` and `missing_modality` hit
**collaborators only**. See `docs/injectors.md`.

`lidar_snow` simulates per beam and takes about 1–3 minutes per frame and agent; it generates its snowflake
particle files once per severity into `$COOPFAULTBENCH_CACHE_DIR` (default `~/.cache/coopfaultbench`).

## Datasets

Nothing from the datasets is redistributed. The paper uses the **V2XSet** test split (19 scenarios, 2,834
frames) and, for CoSDH, the **OPV2V** test split. Both are distributed by their authors; the OpenCOOD
repository (<https://github.com/DerrickXuNu/OpenCOOD>) documents where to download them and how to lay them
out. Check each dataset's terms before use. See `docs/data_access.md`.

## Licence

**Everything is MIT except two bounded directories** (details in `NOTICE`):

- `coopfaultbench/injectors/lidar_weather/` — the LiDAR fog and snowfall simulations adapted from Hahner et
  al. — is **CC BY-NC 4.0: non-commercial use only**. `LidarFogInjector` and `LidarSnowInjector` (and the
  `lidar_fog` / `lidar_snow` fault conditions) run this code and are therefore **non-commercial only**.
- `coopfaultbench/injectors/_vendor/imagecorruptions/` — the camera-corruption backend — is **Apache-2.0**.

The results in `results/` are **CC BY 4.0**. MultiCorrupt's point-reduction code in
`coopfaultbench/injectors/_vendor/multicorrupt/` keeps its own MIT notice. OpenCOOD, CoSDH, V2X-ViT, model
checkpoints and datasets are not included and keep their own terms.

## Citation

Please cite the paper (see `CITATION.cff`). If you use the LiDAR fog or snow injectors, also cite Hahner et
al.'s fog and snowfall simulation papers (listed in their repositories, see
`coopfaultbench/injectors/lidar_weather/NOTICE`) and MultiCorrupt (arXiv:2402.11677).

## More

`docs/` (methodology, injectors, model provenance, data access, reproduction, limitations, results schema),
`CHANGELOG.md`, `ROADMAP.md`.
