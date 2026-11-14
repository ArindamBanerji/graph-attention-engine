# StateBackend protocol design

## Problem

Local file persistence is appropriate for single-process development, but a
read-modify-write cycle across Kubernetes pods or Airflow workers can lose a
concurrent scorer update. Atomic replacement of a complete file prevents a
partial write, not a stale writer from replacing another worker's newer state.

## Protocol

`gae.state.StateBackend` defines four synchronous operations:

```python
load_category(scorer_id, category) -> tuple[dict, str]
save_category(scorer_id, category, state, version) -> str
load_all(scorer_id) -> dict[str, tuple[dict, str]]
list_scorers() -> list[str]
```

`save_category` is a compare-and-swap (CAS) upsert. It accepts only the
version returned by the latest load of that category and otherwise raises
`VersionConflictError`. The returned replacement token is opaque: callers must
compare or supply it, never parse it. A missing category reads as `({}, "0")`;
saving it with version `"0"` creates it.

## Design decisions

- CAS is per category, rather than per complete scorer. Concurrent writes to
  different categories therefore do not conflict.
- Versions are opaque strings so a backend can use counters, ETags, database
  row versions, or another native concurrency token.
- State writes are upserts. The documented empty state and initial token make
  first writes race-safe without a separate create operation.
- On `VersionConflictError`, a scorer caller reloads that category, reapplies
  the same verification to the fresh category state, and retries. It must not
  merge or overwrite centroids from a stale snapshot. CAS produces a valid
  serial order for same-category updates; the current count-decayed and clipped
  scorer updates are not strictly commutative, so the winning write order can
  affect the exact final centroid values.
- The in-memory backend returns defensive copies, so callers cannot mutate its
  stored state without a successful CAS write.

## Backward compatibility

This protocol is additive. P4 will make backend injection optional: no backend
argument keeps the existing local-file persistence behavior and public API
unchanged. Existing single-process users need not configure a scorer ID or a
backend.

## Backend roadmap

1. `InMemoryStateBackend` is the completed reference implementation and shared
   conformance-test target.
2. Redis should implement each category as a key, using `WATCH`/`MULTI` or a
   Lua CAS script.
3. PostgreSQL should store one row per `(scorer_id, category)` and use a version
   predicate in `UPDATE` (or row-version/ETag equivalent).

## Intended scorer integration

P4 will serialize one category slice of the scorer's mutable learning state
(`mu[c, :, :]`, `counts[c, :]`, and category-local learning metadata) and use
the backend at verification time:

```python
state, version = backend.load_category(scorer_id, category)
apply_category_update(state, verification)
try:
    backend.save_category(scorer_id, category, state, version)
except VersionConflictError:
    # Reload this category, reapply this verification, then retry. Do not merge
    # centroid values from the stale snapshot.
    ...
```

This preserves concurrent updates to other categories and leaves the current
local persistence path untouched until backend wiring is introduced.
