# Roadmap

## v0.1.1 (planned)

- Packaging: move the licence metadata to the SPDX string form, `license = "MIT AND CC-BY-NC-4.0 AND Apache-2.0"`
  (the expression `pyproject.toml` already declares), with `license-files`, once
  Python 3.7 / setuptools < 77 no longer has to build the package (setuptools >= 77 warns about the table form,
  with a 2027-02-18 deadline).
- A clean-room rewrite of `plasma_fractal` in the vendored imagecorruptions code (credited there to
  FLHerne/mapgen, licence unresolved; see `coopfaultbench/injectors/_vendor/imagecorruptions/NOTICE`).

## v0.2 (out of scope for v0.1.0)

- A plugin architecture for injectors, datasets and models (registry, entry points).
- Beam reduction for LiDAR (the original MultiCorrupt-based injector did not return its result and is not
  shipped).
- A deterministic noise-floor selection for the LiDAR snow simulation (today it depends on numpy's
  `argpartition` tie order, so outputs differ between numpy versions and CPUs; changing it changes outputs).
- Global-RNG containment for the camera corruptions (they seed numpy's global stream, as MultiCorrupt does).
- More datasets (DAIR-V2X, Griffin) and their loaders.
- A documentation site.
- Model weights are not planned: checkpoints stay with their authors (`docs/model_provenance.md`).

