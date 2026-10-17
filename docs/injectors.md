# Injectors

All injectors are in `coopfaultbench.injectors`. They take plain arrays (an (N, 4) LiDAR array
`[x, y, z, intensity]`, an (H, W, 3) uint8 image, a 4x4 pose) or a `CooperativeSample`, and every random draw
comes from an explicit seed.

## Which agents a fault touches (ego scope)

This matters for interpreting every number in the paper:

| fault | agents touched (paper protocol) | why |
|---|---|---|
| `points_reduce` | **all agents, including the ego** | sensor degradation |
| `lidar_fog` | **all agents, including the ego** | environmental: the fog surrounds the ego too |
| `lidar_snow` | **all agents, including the ego** | environmental |
| `pose_error` | collaborators only | a property of each sender's localisation |
| `latency` | collaborators only | the ego's own data is never stale |
| `agent_drop` | collaborators only | the ego never drops itself |
| `missing_modality` (`missing_lidar`) | collaborators only | a V2X-link fault in the paper's grid |

`FaultSpec` gives `points_reduce`, `lidar_fog` and `lidar_snow` the default scope `'all'`; pass
`'scope': 'non-ego'` to restrict them to collaborators. The ego pose is never perturbed: the OpenCOOD wrapper
raises if it would be, because the ground truth is generated in the ego frame.

## The paper's fault grid

`coopfaultbench/configs/fault_grid.yaml` (also `coopfaultbench.configs.condition_spec(injector, tier)`):

| injector | mild | moderate | severe |
|---|---|---|---|
| pose_error (sigma_xy m / sigma_heading deg) | 0.2 / 0.2 | 0.4 / 0.4 | 0.6 / 0.6 |
| latency (mean delay) | 100 ms | 200 ms | 300 ms |
| agent_drop (p_drop) | 0.25 | 0.50 | 0.75 |
| missing_modality (p_drop_lidar) | 0.25 | 0.50 | 0.75 |
| points_reduce (fraction kept) | 0.30 | 0.20 | 0.10 |
| lidar_fog (MultiCorrupt severity; alpha, beta) | 1 (0.02, 0.008) | 2 (0.03, 0.008) | 3 (0.06, 0.05) |
| lidar_snow (MultiCorrupt severity; snowfall rate mm/h, terminal velocity m/s) | 1 (0.5, 1.2) | 2 (2.5, 1.6) | 3 (1.5, 0.4) |

One base seed (1234) for every cell; each draw is seeded with
`SeedSequence(seed, spawn_key=(frame, crc32(agent), stage))` (`coopfaultbench.faults.derive_seed`), so a
result does not depend on loading order or the number of DataLoader workers.

Labels for the weather tiers: MultiCorrupt describes its fog levels as 300 / 150 / 50 m visibility and its
snow levels as 5 / 35 / 70 mm/h. The values above are the parameters the code actually uses; quote those when
precision matters.

## Notes per injector

- **LiDAR fog / snow (CC BY-NC 4.0, non-commercial only).** The simulations expect intensities on a 0–255
  scale; OPV2V/V2XSet store 0–1, so the wrappers scale to 0–255 for the simulation and back. Snow runs a
  per-beam simulation (about 1–3 minutes per frame and agent) on an HDL-64 sensor model with 64 elevation
  bins; points beyond 118 m pass through unchanged; on OPV2V/V2XSet use `mount_height=1.9`. Its snowflake
  particle files are generated once per severity into `$COOPFAULTBENCH_CACHE_DIR/snow_particles/`
  (default `~/.cache/coopfaultbench/`).
- **Points reduction** keeps a uniform random 30 / 20 / 10 % of the points (MultiCorrupt's `pointsreducing`),
  with numpy's global random state saved and restored around the call.
- **Latency** replaces a collaborator's frame by a stale one, `round(delay / frame period)` frames earlier,
  clamped at the start of the scenario. In OpenCOOD it acts through `time_delay_calculation`, so models that
  consume `time_delay` see it. The injected delay replaces the dataset's own one-frame async delay.
- **Agent drop** is i.i.d. per (frame, collaborator); bursty (Gilbert–Elliott) drop is sequential and is not
  supported by the per-frame seeding.
- **Camera corruptions** (brightness, darkness, fog, snow) need the `camera` extra. They seed numpy's global
  random state, as MultiCorrupt does. They are not used in the paper's evaluation.
- **Sensor occlusion, calibration error, temporal misalignment, bandwidth limit** are included but not used in
  the paper's evaluation.

- **CoSDH path.** `coopfaultbench.opencood.cosdh` applies the faults at CoSDH's own hook points: points
  reduction runs before the weather faults (the OpenCOOD path runs weather first), the pose error is drawn
  with its own normal draws on CoSDH's pose vector rather than through `PoseErrorInjector`, and there is no
  weather cache. For the single-fault cells of the paper the order does not matter.
- **Process-wide side effects.** The fog simulation keeps its random generator at module level (reset from the
  seed on every call), so it is not thread-safe: use one process per worker. Importing the camera backend
  applies its NumPy-2 compatibility shim (`numpy.float_ = numpy.float64`) process-wide.

`injector_fingerprint()` hashes the injector sources; results files record it.
