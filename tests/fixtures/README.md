# Scorer pickle fixture

`v0.7.27_scorer_state.pkl` is a **simulated legacy state dictionary**, generated
with v0.8.0 and the missing-field recipe requested for F8. It is not an artifact
produced by an installed v0.7.27 release. It tests missing-field restoration,
not historical package/class compatibility or downgrade compatibility.

Generation: build a scorer with categories `routing, validation`, actions
`approve, flag, escalate`, and factors `freshness, quality, reliability`.
Apply 20 updates to category 0 with `[0.5, 0.3, 0.8]`, action `i % 3`, and
correctness `i % 4 != 0`; then 15 updates to category 1 with `[0.7, 0.2, 0.6]`,
action `i % 3`, and correctness `i % 3 != 0`. From `__getstate__()`, remove
`_verified_count`, `_correct_count`, `_centroid_drift`, `_backend`, `_scorer_id`,
`_backend_versions`, `factors`, and `_factors`. Serialize that dictionary with
`pickle.HIGHEST_PROTOCOL`. The fixture is checked in so tests do not regenerate
it using the code they are testing. Load only this trusted repository fixture.
