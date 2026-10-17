# Changelog

## 0.1.0 (2026-10-19)

First public release, accompanying the WACV 2027 paper "Agent-Dependent Ground Truth Biases Robustness
Evaluation in Collaborative Perception".

- Fault injectors (cooperative, LiDAR, camera, sensor dropout) with per-(frame, agent, stage) seeding. On 33
  synthetic cases they give byte-identical outputs to the original code under the same numpy version, checked
  under numpy 1.21.6 (the environment of the paper's fault runs; the CoSDH runs used 1.21.5) and 2.5.1
  (`tests/test_injector_golden.py`).
  LiDAR snow output depends on the numpy version and, for numpy 2, on the CPU (`docs/limitations.md`).
- LiDAR snow: two small helpers of unknown licence in the snow code as received via MultiCorrupt (ideas originally
  from a GitHub gist and a GeeksforGeeks page) are replaced by MIT reimplementations written from a description of
  what they compute (`injectors/_snow_helpers.py`), verified byte-identical.
- Licence boundary: the Hahner et al. fog/snow code in `injectors/lidar_weather/` (CC BY-NC 4.0), the
  image-corruption backend in `injectors/_vendor/imagecorruptions/` (Apache-2.0), and MultiCorrupt's
  point reduction in `injectors/_vendor/multicorrupt/` (MIT).
- The fog lookup tables ship as `.npz` arrays (same values as the original pickle files).
- Snowflake particle files are generated on first use into a cache directory (seed 42 + channel); the
  generator reproduces the 192 files used for the paper byte for byte, under numpy 1.21.6 and 2.5.1.
- Dual-reference (fixed vs union) evaluation, arrival-order VOC AP, Monte Carlo agent drop (reproduces the
  paper's agent-drop payload exactly from its rows).
- Five-family fault sweep: OpenCOOD driver and the scene-bootstrap / Holm analysis. On the conditions whose
  rows were available for checking (CoBEVT: clean; points reduction, severe) every condition-level and
  per-scene value of the paper's payload is reproduced exactly; Holm flags (which depend on all 16 conditions)
  were not part of that check.
- CoSDH driver (`scripts/run_cosdh_inference.py`).
- OpenCOOD integration: fault injection inside OpenCOOD's unmodified inference; OpenCOOD's own `x_to_world`
  is used; no OpenCOOD code is included.
- The paper's result payloads, tables and figures, with scripts that rebuild them on a CPU.
- Payloads produced after the provenance tag for the camera-ready version: scene-level bootstrap intervals for the agent-drop cells (with every replicate) and the reference-set shrinkage under agent drop (`results/README.md`).
