# Known limitations

- **One test split per model family.** V2XSet test (19 scenarios) and OPV2V test; scene-level uncertainty
  comes from 19 scenes, and the 4- and 5-agent strata contain only 3 and 2 scenes.
- **Monte Carlo ranges are not confidence intervals.** The agent-drop payload's brackets reflect agent draws
  with the scenes fixed.
- **AP depends on frame order** (arrival-order AP, no global re-sort).
- **Bootstrap means vs plug-in values.** The fault-sweep tables print bootstrap means.
- **Weather cells and two code versions.** A cache of weather outputs was added during the original runs;
  about 6 % of the snow frame-visits predate it. The cache was shown to be output-identical on single frames,
  not on every frame.
- **Union values for the five-family sweep were not computed**; that payload is fixed-reference only.
- **CoSDH** runs on a retrained checkpoint (see `docs/model_provenance.md`) and has no agent-drop or
  missing-modality cells.
- **LiDAR snow depends on the numpy version and CPU.** The snow simulation (Hahner et al.) selects its noise-floor
  bin with `numpy.argpartition` over a histogram with many ties; the tie order differs between numpy 1.x and 2.x,
  and in numpy 2 between CPU SIMD levels. On a synthetic 6,000-point cloud (severity 1) the simulation removes
  186 points under numpy 1.21.6, 570 under numpy 2.5.1 on an AVX-512 CPU and 2,755 under numpy 2.5.1 with AVX2
  dispatch. The paper's fault runs used numpy 1.21.6 (CoSDH: 1.21.5, which gives the same snow output); use 1.21.6 to
  reproduce the snow cells. A
  deterministic replacement is planned (ROADMAP) and would change outputs, so v0.1.0 keeps the original code.
- **Figure files are compared by content, not bytes.** Regenerated PDFs have identical content (every object,
  decompressed stream and image pixel) to `results/figures/`, but the compressed bytes of image streams can differ
  across CPU architectures (Pillow's bundled zlib-ng compresses differently on ARM, e.g. Apple Silicon, than on
  x86-64), so the regenerated files' checksums need not match there. `tests/test_rebuild_figures.py` compares the
  content (`tests/pdf_content.py`).
- **GPU nondeterminism**: inference differs between runs by up to about 0.03 m in box corners and 0.014 in
  score.
- **Licences**: the LiDAR fog and snow code is non-commercial (CC BY-NC 4.0). The snow code as received via
  MultiCorrupt carried two small helpers of unknown licence (`ransac_polyfit`, `binary_angle_search`; ideas
  originally from a GitHub gist and a GeeksforGeeks page, cited in `lidar_weather/NOTICE`). They are replaced by
  MIT reimplementations written from a description of what they compute
  (`coopfaultbench/injectors/_snow_helpers.py`), verified byte-identical.
