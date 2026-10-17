# Models and checkpoints

No weights are distributed. The paper ran each model at its authors' (or OpenCOOD's) released configuration,
through the unmodified OpenCOOD-based code. The SHA-256 values below identify the exact files used; check your
download against them.

| model | dataset | code base used | checkpoint file used (SHA-256) |
|---|---|---|---|
| CoBEVT | V2XSet | OpenCOOD (`point_pillar_cobevt`) | `net_epoch60.pth`, 44,998,363 B, `ddb372a33ec7b20fd5b3ecce34c5b8ac1eed9d5f6506c9772ba3a487b1f65af6` |
| V2X-ViT | V2XSet | the V2X-ViT code base (namespace `v2xvit`) for the fault-inference grid; OpenCOOD's V2X-ViT implementation (`point_pillar_transformer`) for the five-family sweep and the agent-subset sweep | `net_epoch60.pth`, 56,839,201 B, `3e6add2bbcd3064eefba36facda31c8aeaa361f8f54c1d1fa87542e1e3061cb2` |
| Where2comm | V2XSet | OpenCOOD (`point_pillar_where2comm`) | `net_epoch50.pth`, 32,310,681 B, `781a034aac21e720093e061c3895c2130171de83e5be7f61808730a311308f31` |
| CoSDH | OPV2V | CoSDH (<https://github.com/Xu2729/CoSDH>, an OpenCOOD fork) | `net_epoch22.pth`, 35,841,807 B, `df8ae2f11681e0a8e2b86be9d82190b723f7b9b0ef6a88a933c1fa21e4f63f4b` |

Code bases used (commit of the local clone, which did not move between cloning and the runs):
OpenCOOD <https://github.com/DerrickXuNu/OpenCOOD> at `31ba16025da27ffe4e336f011290dfbc66f9a1f1`;
V2X-ViT <https://github.com/DerrickXuNu/v2x-vit> at `f0e6c13f41e916548b2d8aba61e42a18ce980416`;
CoSDH <https://github.com/Xu2729/CoSDH> at `b270b69fe133c0c1ca2044ad072cd77fc21863fb`.
Model repositories: CoBEVT <https://github.com/DerrickXuNu/CoBEVT>, Where2comm
<https://github.com/MediaBrain-SJTU/Where2comm>.

Which code base ran which experiment: the fault-inference grid (`results/payloads/fault_inference_grid.csv`,
the per-cell results, the models/reproduction table, the severe-tier ordering and the all-model degradation
figure) ran CoBEVT and Where2comm through OpenCOOD, V2X-ViT through the V2X-ViT code base, and CoSDH through
CoSDH. The five-family fault sweep and the agent-subset sweep ran all three V2XSet models through OpenCOOD.

Where the checkpoints come from:
- **CoBEVT, Where2comm, V2X-ViT (V2XSet):** the V2XSet checkpoints published with the OpenCOOD / V2X-ViT
  benchmark tables. The CoBEVT and Where2comm files are byte-identical to the members of the archives
  `cobevt_lidar.zip` and `point_pillar_where2comm_v2xset.zip` from those releases. The exact public download
  page of each archive is linked from the OpenCOOD and V2X-ViT READMEs; this release could not re-verify the
  links (no network access during preparation).
- **Where2comm on V2XSet is not the Where2comm authors' release.** The Where2comm paper evaluates on OPV2V,
  V2X-Sim, DAIR-V2X and CoPerception-UAVs; the V2XSet checkpoint is a third-party training published as a
  V2XSet baseline. Do not attribute its numbers to the Where2comm authors.
- **CoSDH:** the checkpoint released by the CoSDH authors. Their README states: "These checkpoints have been
  retrained and the results differ slightly from those reported in the paper." The paper's CoSDH row is
  therefore a reproduction with that checkpoint, not a re-measurement of the published numbers.

Each model runs at its own **shipped** `wild_setting` (OpenCOOD's localisation-noise and asynchrony switches,
on in some released configurations as part of their "noisy" protocol); the faults are injected on top, and
the clean control is measured at the same setting. `scripts/run_fault_inference.py --fi_force_wild_off` turns
both switches off instead.

Licences of the model code bases: CoBEVT Apache-2.0; V2X-ViT MIT; Where2comm: no licence file found;
OpenCOOD and CoSDH: academic licences (check them before use). Checkpoints carry no stated terms.
