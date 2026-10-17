# Result payload schemas

This document gives the key structure of every file under `results/payloads/`.
For estimators and caveats, see `results/README.md`. For provenance and
hashes, see `results/MANIFEST.md`.

Notation:
- `<model>` is one of `cobevt`, `v2xvit`, `where2comm`, or `cosdh` (grid
  only).
- `<PER_CONDITION>` and `<PER_SCENE>` stand for the two per-model block keys
  of the fault-sweep payload. These are internal names fixed by the frozen
  payload; they are available as `KEY_PER_CONDITION` and `KEY_PER_SCENE` in
  `coopfaultbench/reporting/payloads.py`.
- `<iou>` is `"0.5"` or `"0.7"`. These are string keys.
- `<tier>` is `mild`, `moderate` or `severe`.
- A "triple" is a 3-element list.

## `fault_sweep_fixed_reference.json`

```
{
  "boot_M": 1000,                     # bootstrap replicates
  "seed": 20260826,                   # SeedSequence seed
  "conditions": [16 strings],         # "clean", "<fault>_<tier>" for fault in
                                      #   pose_error, latency, points_reduce,
                                      #   lidar_fog, lidar_snow
  "models": {
    "<model>": {                      # cobevt, v2xvit, where2comm
      "<PER_CONDITION>": {            # per-condition AP and Gain
        "<iou>": {
          "<condition>": {
            "ap_indiv_ci": [mean, p2.5, p97.5],   # AP, ego-only arm
            "ap_cp_ci":    [mean, p2.5, p97.5],   # AP, full-roster CP arm
            "gain_ci":     [mean, p2.5, p97.5],   # AP_CP - AP_indiv per replicate
            "n_scenarios": 19,
            "p_boot": float,          # two-sided bootstrap p-value proxy
            "uncorrected_reject": bool,   # gain interval excludes 0
            "holm_reject": bool       # and Holm over the 16 conditions
          }
        }
      },
      "<PER_SCENE>": {                # per-scene verdicts
        "<iou>": {
          "<condition>": {
            "<scenario>": {           # 19 V2XSet test scenario names
              "gain_ci": [mean, p2.5, p97.5] | null,
              "verdict": "established beneficial" | "not established"
                         | "established harmful" | "N/A",
              "uncorrected_reject": bool,
              "holm_reject": bool     # Holm over the 19 scenes
            }
          }
        }
      }
    }
  },
  "triggers": [                       # every "established harmful" verdict
    {"rule": str,                     # one rule name for all 209 entries:
                                      #   per-scene harmful verdict
     "model": str, "iou": str, "condition": str,
     "scenario": str, "gain_ci": [mean, p2.5, p97.5]}
  ],
  "halted_cells": []                  # (model, condition) cells not aggregated
}
```

The tables and figures read `ap_indiv_ci[0]`, `ap_cp_ci[0]` and `gain_ci`
(the bootstrap mean and percentile bounds), plus `uncorrected_reject`,
`holm_reject` and the per-scene `verdict` strings.

## `agent_drop_fixed_vs_union.json`

```
{
  "M": 100,                           # Monte Carlo agent-drop draws
  "seed": 20260826,
  "tiers": {"none": 0.0, "mild": 0.25, "moderate": 0.5, "severe": 0.75},
  "models": {
    "<model>": {                      # cobevt, v2xvit, where2comm
      "n_frames": 2834,
      "tiers": {
        "<none|mild|moderate|severe>": {
          "<iou>": {
            "mean": float,            # mean over draws of dAP = AP_fixed - AP_union
            "std": float,             # std over draws of dAP
            "ci": [p2.5, p97.5],      # Monte Carlo agent-draw RANGE of dAP
                                      #   (NOT a confidence interval)
            "ap_union": {"mean": float, "ci": [p2.5, p97.5]},
            "ap_fixed": {"mean": float, "ci": [p2.5, p97.5]},
            "by_stratum": {           # by scene agent count; not used by
                                      #   any released table or figure
              "<2|3|4|5>": {"mean": float, "std": float,
                            "M": int}  # draws contributing (100 everywhere)
            }
          }
        }
      }
    }
  }
}
```

## `agent_drop_scene_bootstrap.json`

```
{
  "meta": {"B": 1000, "seed": 20260826, "n_frames": int,
           "identity_values_compared": int,   # published values the job reproduced exactly
           "strata": {"<2|3|4|5>": {"n_scenes": int, "scenes": [str], "n_frames": int}},
           "script_sha256": str, "prereg_sha256": str, "rows_list_sha256": str, "n_row_files": int,
           "numpy": str, "python": str},
  "gates": {"inputs": "PASS", "identity": "PASS", "coverage": "PASS", "null_control": "PASS"},
  "cells": {
    "<model>|<tier>|<iou>": {         # 3 models x mild/moderate/severe x 0.5/0.7 = 18 cells
      "model": str, "tier": str, "iou": str, "p_drop": float,
      "plugin_mc_mean": float,        # the published dAP mean (agent_drop_fixed_vs_union.json)
      "mc_agent_draw_range": [p2.5, p97.5],  # the published Monte Carlo agent-draw range
      "plugin_ap_union": float, "plugin_ap_fixed": float,  # the published AP means
      "plugin_rel": float,            # mean over the 100 draws of each draw's (AP_fixed - AP_union) / AP_union;
                                      #   not a published value, and not the ratio of the published means
      "primary": {"boot_mean": float, "ci": [lo, hi], "excludes_zero": bool,
                  "verdict": "established" | "not established",
                  "p_boot": float, "holm_reject_18": bool},
      "frame_weighted": {"boot_mean": float, "ci": [lo, hi], "excludes_zero": bool,
                         "plugin_mc_mean": float},
      "unstratified": {"boot_mean": float, "ci": [lo, hi], "excludes_zero": bool},
      "rel": {"boot_mean": float, "ci": [lo, hi]},
      "ap_union": {"boot_mean": float, "ci": [lo, hi]},
      "ap_fixed": {"boot_mean": float, "ci": [lo, hi]},
      "per_stratum": {
        "<2|3|4|5>": {"n_scenes": int, "sound": bool,
                      "ci_status": "printable" | "not interpretable: too few scenes",
                      "boot_mean": float, "ci": [lo, hi],             # dAP
                      "plugin_mc_mean": {"dAP": float, "ap_union": float, "ap_fixed": float, "rel": float},
                                      # means over the 100 draws; "rel" as for plugin_rel
                      "rel": {...}, "ap_union": {...}, "ap_fixed": {...}}  # boot_mean, ci, ci_status
      }
    }
  }
}
```

## `reference_shrinkage.json`

```
{
  "meta": {"M": 100, "script_sha256": str, "numpy": str, "python": str},
  "gates": {"inputs": "PASS", "identity": "PASS", "null_control": "PASS"},
  "models": {
    "<model>": {                      # cobevt, v2xvit, where2comm
      "n_frames": int,
      "frames_per_stratum": {"<2|3|4|5>": int},
      "tiers": {
        "<mild|moderate|severe>": {
          "p_drop": float,
          "pooled": {"mean": float, "std": float, "range_2.5_97.5": [p2.5, p97.5]},  # over the M draws
          "per_stratum": {"<2|3|4|5>": {"mean": float, "std": float, "range_2.5_97.5": [p2.5, p97.5]}}
        }
      },
      "ego_only": {"pooled": float, "per_stratum": {"<2|3|4|5>": float},
                   "frames_with_no_collaborator_in_range": int},
      "geometry": {"n_frames": int, "frac_frames_with_a_cav_cut": float,
                   "r_pooled": float, "r_within_stratum": float,
                   "r_by_stratum": {"<N>": {"r": float, "n_frames": int}}}
    }
  }
}
```
Shrinkage values are fractions (-0.05 = 5 % of the fixed reference's boxes leave the union reference).

## `fault_inference_grid.csv`

The file has 109 data rows, comma-separated, with CRLF line ends and quoting
as written by Python's `csv` module.

| Column | Meaning |
|---|---|
| `model`, `dataset`, `setting` | model id; `v2xset` or `opv2v`; shipped evaluation setting (`perfect` / `noisy`) |
| `injector`, `severity_tier` | `none`/`clean`, or injector and `mild`/`moderate`/`severe` |
| `severity_value`, `severity_unit` | injector parameter at that tier (e.g. `0.2m/0.2deg`, `pose_sigma`) |
| `seed` | evaluation seed (1234) |
| `clean_ap50`, `clean_ap70` | clean AP of the model (per stratum for agent drop) |
| `faulty_ap50`, `faulty_ap70` | AP under the fault, 6 dp |
| `delta_ap50`, `delta_ap70` | faulty - clean |
| `rel_drop_ap50`, `rel_drop_ap70` | delta / clean |
| `n_injections`, `n_frames` | injection count; frames evaluated (per stratum for agent drop) |
| `determinism_floor_ap50`, `determinism_floor_ap70` | clean vs. clean-repeat AP difference of that model |
| `extra_metric` | `;`- or `,`-separated `key=value` diagnostics (e.g. `scene_agent_count=2,mean_dropped=0.230,n_scenes=7`, `removed_frac=0.3991`) |
| `timestamp` | completion time of the cell |

## `cells/<model>/<injector>/<tier>.json`

Each file is one JSON object. Key order is as in the original, and the file
is written with `indent=1`.

The keys present depend on the model:

V2XSet models (CoBEVT, V2X-ViT, Where2comm)

| Key | Meaning |
|---|---|
| `condition`, `spec_json`, `spec`, `seed` | fault specification of the cell |
| `wild_setting_shipped` | shipped evaluation noise setting: `loc_err`, `async`, `xyz_std`, `ryp_std`, `async_overhead` |
| `wild_forced_off` | flag |
| `n_frames` | frames evaluated |
| `ap_30`, `ap_50`, `ap_70` | full-precision AP |
| `strata` | per scene-agent-count `{ap_50, ap_70, n_frames, n_scenes}`, present on every V2XSet cell (the grid reports it for agent-drop rows) |
| `wrapper_contract_version`, `adapter_sha256`, `injector_contract_version`, `injector_sha256`, `injector_files_sha256` | code fingerprints (only on cells run after fingerprinting existed) |

CoSDH

| Key | Meaning |
|---|---|
| `ap_30`, `ap_50`, `ap_70`, `n_gt` | AP and ground-truth count |
| `seed`, `spec_json` (and `spec` on the re-stamped `none/` cells) | fault specification |
| `fusion_method` | `intermediatelate` |
| `cosdh_contract_version`, `cosdh_adapter_sha256` | adapter fingerprint |
| `wrapper_contract_version`, `adapter_sha256`, `injector_*` | on the two re-stamped `none/` cells only |

Four keys were removed from every cell: `model_dir`, `opencood_path`, `job_id`
and `frame_stratum`.

## `fault_inference_models.json`

```
{
  "models": {
    "<model>": {
      "dataset": "v2xset" | "opv2v",
      "setting": "perfect" | "noisy",
      "expected_frames": int,
      "graded": bool,                 # as recorded in the grid manifest
      "published": {"ap50": float, "ap70": float} | null
    }
  },
  "not_applicable_cells": ["cosdh/agent_drop/mild", ...]   # 6 entries
}
```
