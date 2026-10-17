# Data access

CoopFaultBench distributes **no dataset content**: no point clouds, no images, no ground-truth boxes, and no
per-frame rows derived from them. `results/` holds aggregates only (APs, intervals, per-condition and
per-stratum summaries).

- **V2XSet** (used for CoBEVT, V2X-ViT, Where2comm): the test split, 19 scenarios, 2,834 frames.
- **OPV2V** (used for CoSDH): the test split, 2,170 frames.

Both datasets are distributed by their authors; the OpenCOOD repository (<https://github.com/DerrickXuNu/OpenCOOD>)
documents where to download them and the expected directory layout. We found no explicit terms of use for
V2XSet; the OPV2V website states "All Rights Reserved". Check the terms yourself before using or sharing
anything derived from them.

Point each checkpoint's `config.yaml` (`root_dir`, `validate_dir`) at your copy of the test split. The seeds
are keyed on the frame index, so reproducing the paper's numbers needs the identical test split and frame
order.
