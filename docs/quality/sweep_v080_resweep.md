# GAE v0.8.0 verification re-sweep

Reviewer: gpt-6-astra. Date: 2026-11-14.
Scope: original F1–F7 reproductions, six audit dimensions, changes introduced by their fixes.
Source files modified: **0**. This requested report is the only file created.

## Original Finding Verification (F1–F7: each FIXED or REGRESSION)

The supplied scenarios were executed as Python stdin programs, using the same input values, update counts, scorer configurations, exception injection and assertions. Shell quoting was adapted for PowerShell. All seven assertions passed. The original report and the session state's SWEEP FIX entry were read first.

| Finding | Status | Observed result | Fixed-code reference |
|---|---|---|---|
| F1: stale save overwrites newer state | **FIXED** | Backend verified_count remains **11** after S.save(), despite S previously holding 10 | profile_scorer.py:1421–1444 |
| F2: fresh scorer ignores backend | **FIXED** | New scorer has **50 verified**, selects **action 1** | profile_scorer.py:385–386; factory 1708–1764 |
| F3: action-axis mismatch | **FIXED** | Construction rejects reversed actions with **ValueError: Action axis mismatch for category 0: stored=['x', 'y'], current=['y', 'x']** | profile_scorer.py:847–860 |
| F4: mutation after failed write | **FIXED** | Injected OSError propagates; local verified_count stays **1 → 1** | profile_scorer.py:1017–1036 |
| F5: categories compared by position | **FIXED** | **rolled_back**, deltas **{'a': -1.0, 'b': 1.0}** | agentevolver.py:112–133 |
| F6: direct result maps mutable | **FIXED** | Assigning base_accuracy['a'] raises **TypeError** | agentevolver.py:36–39 |
| F7: zero timeout | **FIXED** | Sufficient-data candidate returns **timeout** at timeout_seconds=0 | agentevolver.py:86 |

### Execution traces

**F1:** S.update stores both the category state and returned version; U auto-loads that category and advances it to count 11. S.save uses S's cached version at 1427, so the first CAS fails. The conflict branch reads U's state at 1434–1436, applies it at 1437, updates the cached token at 1438, and retries with the newly extracted count 11. There is no borrow-a-fresh-version/write-an-old-snapshot path. Four attempts maximum are enforced.

**F2:** build_profile_scorer passes backend, factors and load_backend through init_from_config into ProfileScorer. After runtime fields are initialized, constructor line 385 calls load when backend is set and load_backend is true. Thus the tested new scorer restores the 50 verified outcomes before score reads its centroids. Airflow Score and Verify both call this factory (airflow.py:132,189); the full suite includes a real-scorer operator regression covering persistence across those executions.

**F3:** every extracted category now contains _schema. _apply_category_state reads that metadata before assigning centroids; stored ordered actions differ from the new actions, so it raises before applying the category. Failure during construction is the intended consequence of auto-load. Explicit-load rejection remains available with load_backend=False.

**F4:** update loads the authoritative category and extracts a detached list/dictionary snapshot before _update_local. The latter increments local state; injected save failure enters the broad exception handler, which reapplies the snapshot and restores runtime counters/buffer before re-raising. The focused regression also checks centroids, counts, backend state and a subsequent successful update. Conflict attempts restore their speculative mutations before replay as well.

**F5:** category-map construction validates unique equal name sets, then stores each name's separate base and candidate index. Base a=100% and candidate a=0% are compared as the same category even though their tensor positions differ. Delta a=-1 makes ALL_CATEGORIES fail. Candidate counts and WEIGHTED_AVERAGE weights use candidate indices from the same map.

**F6:** __post_init__ copies each supplied map into its own dict and wraps it in MappingProxyType. This applies to direct construction as well as evaluate-produced results. Both dictionary mutation and frozen-field reassignment are covered by the passing regression suite.

**F7:** elapsed >= timeout now handles an equal clock tick. The explicit zero-timeout probe passed; the regression suite also fixes the clock to the same value for construction and evaluation.

## New Issues Introduced by Fixes (if any)

**New P1 findings: 0 found in this re-sweep.**
**New P2 findings: 1.**

### R1 — P2: wrapper factor names are not forwarded to the new scorer metadata

**A — Locate:** JudgmentMemoryCallback.__init__ in gae/integrations/langchain.py:67 and JudgmentMemoryService.__init__ in gae/integrations/mcp.py:48. Factory calls at langchain.py:99–104 and mcp.py:75–80.

**B — Read:** Both constructors and their complete persistence paths were inspected, together with the new ProfileScorer factor initialization and _schema extraction.

**C — Trace:**

1. Construct either wrapper with factors=['freshness','quality'].
2. Wrapper stores those correct names in self.factors.
3. The wrapper's build_profile_scorer call passes categories/actions/neutral centroids/factor count positionally, but omits factors=self.factors.
4. New ProfileScorer initialization at profile_scorer.py:248–250 therefore supplies generated positional names.
5. _extract_category_state at 820–826 writes those generated names into the new backend schema.

Observed for **both** wrappers:

```
wrapper factors ['freshness', 'quality']
scorer factors ['factor_0', 'factor_1']
schema factors ['factor_0', 'factor_1']
```

**D — Verdict:** The wrapper's public domain factor metadata and its exposed scorer's new factor metadata disagree. Existing wrapper JSON save/load still works, because it persists and validates wrapper self.factors separately. Neither wrapper currently exposes a backend constructor parameter. Therefore this is not silent corruption on their supported persistence path, and does not reopen F3's direct-backend failure.

**E — Classification:** P2 integration/metadata inconsistency newly exposed by adding semantic factor names to ProfileScorer. It can impede migrating wrapper scorer state into a backend that correctly uses those named factors. Recommended follow-up: pass factors=self.factors from both wrappers and add a metadata consistency regression. No fix was made during this read-only task.

### Existing limitations retained, not counted as new findings

- Original **F8** remains a P2 validation gap: actual v0.7.27-produced pickle loading and downgrade behavior have not been verified with historical release artifacts. Synthetic missing-field tests and current pickle round trips pass.
- Unknown-commit network failures are re-raised; callers must reconcile durable state before resubmitting. These changes do not provide exactly-once verification.
- Multi-category save is not an atomic transaction across all categories.
- No live PostgreSQL transaction/concurrency test or real Airflow scheduler deployment was run. The PostgreSQL unit tests use a fake; Airflow can use its fallback BaseOperator.
- Legacy records without _schema cannot establish historical axis identities; accepting them is the explicit compatibility policy.

## 6-Dimension Re-Audit Results

### A. API contract stability — PASS

Runtime and source checks confirm:

- ProfileScorer's added factors and load_backend arguments are keyword-only, following the existing backend/scorer_id arguments.
- init_from_config and build_profile_scorer append new options; prior positional arguments retain their order.
- _backend_versions and _schema are internal state/record fields, not additional positional API arguments.
- accuracy, verified_count and centroid_drift still take integer category indices. AgentEvolver and RGIMetrics pass integers. conservation_status remains a property.
- load_backend=False successfully constructs a neutral scorer at verified_count=0; explicit load() restores verified_count=10 in the probe.
- There are no methods named _snapshot_category or _restore_category. The implementation reuses **private, typed** _extract_category_state(self, category_index: int) -> Dict and _apply_category_state(self, category_index: int, state: Dict) -> None, plus private runtime snapshot/restore helpers.
- The factory document explains auto-loading, deferred loading and positional factor-name defaults at profile_scorer.py:1733–1738.

The earlier AST audit found no invalid factor_vector= update calls. The fixes preserve the update(f,...) contract, and the entire existing suite passes.

### B. State serialization safety — PASS for normal scorer state

Re-ran the original ten-update probe with two categories, two actions and two factors. JSON serialization succeeded for the persisted category:

```
JSON a 243
_schema = {
  'actions': ['x', 'y'],
  'factors': ['f1', 'f2'],
  'categories': ['a', 'b']
}
```

_schema is unconditionally emitted by _extract_category_state (profile_scorer.py:820). It consists of lists of names. Centroids/counts/DK/covariance arrays still use tolist(); verified/correct counters and drift use Python scalar values. The new metadata introduces no NumPy serialization problem.

Backend version tokens are kept separately and excluded from pickle __getstate__ (1495); __setstate__ initializes their cache afresh (1576). Newly saved records have a schema; old untouched records may legitimately lack one.

### C. Cross-integration consistency — PASS with R1 note

| Path | Restoration/persistence behavior after fixes |
|---|---|
| Direct ProfileScorer / factory with backend | Auto-load enabled by default; update uses category CAS; explicit opt-out available |
| Airflow Score | Builds scorer through factory, receives restored centroids before scoring |
| Airflow Verify | Builds scorer through factory, then update reads/replays against fresh category state |
| LangChain callback | Constructor still has no backend argument; optional state_path loads its existing JSON format |
| MCP service | Same: JSON state_path restoration, no backend constructor argument |

LangChain and MCP do **not** consume backend _schema, nor do their own JSON files need to add it. Their existing metadata includes categories, actions and factors and validates each axis before accepting compatible saved data (langchain.py:355–375; mcp.py:334–353). Both retain scorer_observability and restore it. The passing suite includes LangChain↔MCP state compatibility. R1 describes the narrower inconsistency in the newly added scorer-level factor-name field.

Airflow gains backend auto-load through the factory; it has not gained the wrappers' JSON file format. No claim of Airflow↔LangChain file compatibility is made.

### D. Optional dependency isolation — PASS

Executed fresh subprocesses that actually blocked imports:

| Allowed/blocked configuration | Core imports and from gae.state import * |
|---|---|
| All six optional packages blocked | PASS |
| Prometheus available; Redis/psycopg2/Airflow blocked | PASS |
| Redis available; Prometheus/psycopg2/Airflow blocked | PASS |
| Locally installed dependencies allowed | PASS |

The fixes introduce no new third-party import. MappingProxyType is standard library and was already used in AgentEvolver. Existing optional imports remain guarded in gae/__init__.py and gae/state/__init__.py. pyproject.toml still declares only NumPy as a base dependency.

### E. Backward compatibility — PASS for requested pre-fix state/deferred-load checks

The probe removed _schema from a stored category, saved it through the backend CAS API, then constructed a new scorer. It loaded count=10 successfully.

Trace: _apply_category_state line 849 reads state.get('_schema'); missing returns None; the validation block is skipped; existing arrays/counters restore normally. Later normal writes emit the new schema. A reversed named axis is correctly rejected when schema is present.

The deferred-load probe returned zero verified before load and ten after. Both keyword names are forwarded through the factory. Existing no-backend behavior and pickle round trips pass in the full suite. Historical missing factors get generated positional names in __setstate__, profile_scorer.py:1577–1578. F8's actual old-package fixture coverage remains outstanding.

### F. AgentEvolver edge cases — PASS for requested cases

- Reordered same-name categories: original regression now rolls back with the correct per-name deltas.
- Equal names/equal order: existing all-category and weighted tests pass.
- Single category: independently exercised ALL_CATEGORIES, WEIGHTED_AVERAGE and ANY_IMPROVEMENT with base 70%, candidate 80%; all returned promoted.
- Different name sets and duplicate names: constructor map validation rejects them; mismatched-name regression passes.
- Unnamed direct scorers retain positional cat_i labels; named callers receive semantic alignment.
- Zero candidate evidence still produces waiting; category-free comparison cannot pass the positive min_decisions gate.
- Zero timeout produces timeout, including an equal monotonic-clock tick.
- Direct EvaluationResult maps are copied and immutable; caller source-dict changes cannot mutate the result.
- History exposes a list copy; produced entries retain immutable maps.

## Full Suite + Mypy

Commands executed:

```text
python -m pytest tests/ -q --timeout=120 --disable-warnings
python -m mypy gae/profile_scorer.py gae/agentevolver.py gae/metrics.py gae/state/ gae/integrations/ --config-file pyproject.toml
```

Results:

```text
1372 passed, 1 skipped, 13482 warnings in 10.27s
Success: no issues found in 13 source files
```

Both exited 0. --disable-warnings suppresses warning details, not test execution or warning counts. No tests or source were edited for this review.

## Release Recommendation: TAG

**TAG with the documented P2 follow-ups.** All five original P1 reproductions and both original P2 reproductions are fixed. No additional P1 was found in the requested six-dimension verification. The new wrapper factor-metadata inconsistency does not break their supported JSON persistence and is non-blocking.

Retain R1 and original F8 in the follow-up backlog. This recommendation is based on the inspected code, deterministic probes and unit/type gates; it is not certification of untested live backend/framework deployment behavior. No tag or publication was performed.

v0.8.0 re-sweep complete.
Original P1s verified fixed: 5/5.
Original P2s verified fixed: 2/2.
New P1 findings: 0.
New P2 findings: 1.
Suite: 1,372 passed, 1 skipped.
Mypy: clean (13 source files).
Release recommendation: TAG.

