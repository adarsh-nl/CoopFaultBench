# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The CoopFaultBench Authors. See LICENSE.
"""Names fixed by the published fault-sweep payload (results/payloads/fault_sweep_fixed_reference.json).

The payload ships byte-identical (its SHA-256 is the provenance anchor), so its block keys and trigger-rule
strings keep the names they were written with. They are short labels from the original analysis:
'q1_q2' = per-condition cooperative gain with Holm correction, 'q3' = per-scene verdicts. Code that reads
or writes the payload uses these constants; nothing else in the package uses the labels.
"""

#: per-condition block: cooperative gain, intervals, p-value proxy, Holm flags
KEY_PER_CONDITION = 'q1_q2'
#: per-scene block: per-scene gain interval and verdict
KEY_PER_SCENE = 'q3'
#: trigger rule: a scene whose gain interval excludes zero on the harmful side
RULE_SCENE_HARMFUL = 'Q3-harmful'
#: trigger rule: the clean condition's gain is not positive and its interval excludes zero
RULE_CLEAN_GAIN_NOT_POSITIVE = 'Q1-clean-collapse'
