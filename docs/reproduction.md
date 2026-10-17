# Reproduction

## Tables and figures (CPU, minutes)

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
(cd results && sha256sum -c SHA256SUMS)        # verify the shipped files first
pytest                                         # includes the table and figure rebuild gates
python scripts/make_tables.py --out /tmp/cfb_tables && python scripts/make_figures.py --out /tmp/cfb_figures
diff -r /tmp/cfb_tables results/tables         # see results/README.md for the one documented difference
```

The table gate regenerates every table and requires byte equality with `results/tables/`. One cell pair differs
by design: the shipped models/reproduction table prints "n/a" for Where2comm's published AP, as in the paper,
while the payload's model record carries values for it (`results/README.md`, "Known mismatch"). With
matplotlib 3.11.1 the figures regenerate with identical content (every object, decompressed stream and image pixel);
their compressed bytes can differ across CPU architectures (Pillow/zlib-ng, e.g. Apple Silicon vs x86-64), so the
figure gate compares content, not bytes. The shipped PDFs equal the paper's, except that the per-scene verdict
composition figure has a reworded title.

## The injectors

`tests/test_injector_golden.py` compares the injectors with outputs recorded from the original code: 33
synthetic cases covering all paper fault families and the camera corruptions. The 30 non-snow cases are
byte-identical under numpy 1.21.6 (Python 3.7, the environment of the paper's fault runs; CoSDH used 1.21.5) and numpy 2.5.1 and
are checked on any numpy. The LiDAR snow output depends on numpy's `argpartition` tie order, which differs
between numpy 1.x and 2.x and, for numpy 2, between CPU SIMD levels; its golden digests are recorded for numpy
1.21.6 and for numpy 2.5.1 on an AVX-512 CPU and are checked only there (marked slow: `pytest -m slow`). To
reproduce the paper's snow cells, use numpy 1.21.x (OpenCOOD's Python 3.7 environment).

## The fault-inference grid and agent drop (GPU, OpenCOOD, data)

See the README: `scripts/run_fault_inference.py` for each grid cell, `scripts/run_agent_subset_sweep.py` +
`scripts/resample_agent_drop.py` for the dual-reference agent-drop payload. `resample_agent_drop.py
--compare results/payloads/agent_drop_fixed_vs_union.json` reports any difference from the paper's payload.
GPU inference is not bit-reproducible between runs, so a fresh inference run reproduces the payload to within
inference noise, not bit for bit.

Approximate costs from the original runs: a fault-inference cell takes about 25 minutes on one GPU, a snow
cell about 3 hours (the snow simulation runs on the CPU).

`scripts/analyze_fault_sweep.py`, run on the paper's cached rows for CoBEVT (clean and points reduction, severe),
reproduces every condition-level and per-scene value of `results/payloads/fault_sweep_fixed_reference.json` for
those conditions exactly (100 of 100 values; checked during release preparation).

`scripts/resample_agent_drop.py`, run on the paper's cached agent-subset rows (not distributed), reproduces
`results/payloads/agent_drop_fixed_vs_union.json` exactly: every value, 0 differences (checked during release
preparation with numpy 2.5.1).
