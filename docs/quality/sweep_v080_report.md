# GAE v0.8.0 quality sweep

Date: 2026-11-14. Reviewer: gpt-6-astra. Scope: six requested dimensions; read-only source audit.

## Summary

**HOLD. Five P1 failures were reproduced despite 1,358 passing tests and clean mypy.** The failures concern lost persisted learning, scoring from fresh rather than persisted state, missing model-axis identity, local mutation after failed persistence, and incorrect comparison of differently named categories.

Additional findings: **3 P2, 0 P3**. The suspected ordinary NumPy-to-JSON serialization failure did **not** reproduce. Missing optional dependencies did **not** break core imports.

Method: `../copilot-sdk/docs/quality/bug_hunt_v5.md`, mandatory steps A–E. Its instruction to stop expanding the hunt at five P1 findings was observed. The six requested dimensions are represented below, but this is not an exhaustive certification of all concurrency, live-database, or framework behavior.

Read project conventions first. Existing CLAUDE/AGENTS text still describes a NumPy-only architecture; explicit v0.8.0 extras supersede that historical description for this task. No source changes, fixes, git operations, installs, external database writes, tags, or publication occurred. One requested Markdown report was created.

Primary source files analyzed (12): `gae/__init__.py`, `gae/profile_scorer.py`, `gae/metrics.py`, `gae/agentevolver.py`, `gae/state/{__init__,protocol,memory_backend,redis_backend,pg_backend}.py`, and `gae/integrations/{langchain,mcp,airflow}.py`. Supporting reads included pyproject, session notes, tests, and category phase definitions.

Verification in this checkout:

- Full suite: **1,358 passed, 1 skipped, 13,454 warnings in 21.16s**, process exit 0.
- Requested mypy coverage: **Success: no issues found in 12 source files**.
- Read-only Python probes reproduced all five P1s and the direct-result mutability issue.
- Import matrix ran in separate subprocesses with actual import blocking; merely assigning a list of blocked dependencies would not test isolation.
- PostgreSQL testing remains fake-backed; this sweep did not validate a live server or real transaction interleavings.

## API Contract Matrix

Runtime `inspect.signature` inspection confirmed the following. References are relative to this repository.

| API | Actual contract | Consumer consistency |
|---|---|---|
| ProfileScorer.accuracy | `accuracy(self, category_index: int) -> float` (793) | AgentEvolver and metrics pass integer indices |
| ProfileScorer.verified_count | `verified_count(self, category_index: int) -> int` (787) | Same |
| ProfileScorer.centroid_drift | `centroid_drift(self, category_index: int) -> float` (800) | Metrics passes integer indices |
| ProfileScorer.conservation_status | Read-only property returning str (772) | Both consumers read the property; neither incorrectly invokes it |
| ProfileScorer.update | `update(self, f: np.ndarray, category_index: int, action_index: int, correct: bool, gt_action_index: Optional[int] = None, confidence: Optional[float] = None) -> CentroidUpdate` (954) | All scanned gae/tests call sites avoid the invalid factor_vector keyword |
| ProfileScorer persistence | `save(self, path=None) -> Optional[Path]`; `load(self, path=None) -> None` (1408,1424) | Semantic defects below; signatures themselves align |
| build_profile_scorer | Includes `backend: Optional[StateBackend] = None`, `scorer_id: str = "default"`, `factors: Optional[List[str]] = None` (1647) | Airflow forwards config verbatim, then factory passes backend/scorer_id through init_from_config (1556,1635) |
| StateBackend / InMemory | `load_category(self, scorer_id: str, category: str) -> tuple[dict,str]`; `save_category(self, scorer_id: str, category: str, state: dict, version: str) -> str`; `load_all(self, scorer_id: str) -> dict[str,tuple[dict,str]]`; `list_scorers(self) -> list[str]` | Redis and PostgreSQL implement the same four operations |
| AgentEvolver | `evaluate(self) -> EvaluationResult`; history property returns a list copy (71,99) | Consumes public integer-index observability |
| RGIMetrics | `update(self) -> None`; `start(self, port: int = 9090) -> None`; `score_timer(self) -> Iterator[None]`; `set_agentevolver_phase(self, variant: str, phase: str = "Shadow") -> None` | Calls correct scorer names |

A2 trace: metrics.py:105–118 iterates integer indices, then calls verified_count/accuracy/centroid_drift; agentevolver.py:75,110–113 does likewise. profile_scorer.py:789 checks integer bounds, and 795–798 returns 0.0 for zero verified, otherwise correct/verified. **No name/type drift.** These APIs do not support category-name strings; backend category strings are a different interface.

A4: AST-scanned every Python file under gae/ and tests/ for update calls carrying a `factor_vector=` keyword, including multiline calls: **zero matches**. Positional variables named factor_vector are valid.

## Serialization Safety Results

Source: ProfileScorer._extract_category_state, profile_scorer.py:806–827.

| Stored key | Normal emitted Python type | Conversion |
|---|---|---|
| centroids | list[list[float]] | ndarray.tolist(), line 809 |
| counts | list[int] | ndarray.tolist(), line 810 |
| verified_count | int | Python counter initialized to 0, incremented at 782 |
| correct_count | int | Python counter incremented at 784 |
| centroid_drift | float | float conversion at 785 |
| category_state.phase | str | CategoryState phase |
| category_state.n_decisions | int | Python counter |
| category_state.freeze_point | int or None | CategoryState freeze point |
| dk_weights (optional) | list[float] | tolist(), line 823 |
| cov_inv (optional) | nested list[float] | tolist(), line 826 |

B1–B4 trace: update → _record_verification → _extract_category_state → backend.save_category. InMemory deep-copies the dictionary; Redis json.dumps at redis_backend.py:74 serializes it; PostgreSQL wraps it in psycopg2.extras.Json at pg_backend.py:71/96. Arrays are converted before the persistence boundary; optional phase state is flattened into a dictionary, not stored as a dataclass.

Executed ten updates against InMemory and successfully serialized load_all with json.dumps:

```
{"a": [{"centroids": [[0.5], [0.55]], "counts": [0, 10],
"verified_count": 10, "correct_count": 10, "centroid_drift": 0.005}, "10"]}
```

The supplied synthetic JSON checks returned: tolist array OK; np.float64 drift OK in this environment; arbitrary np.int64 counter rejected with `Object of type int64 is not JSON serializable`. That last synthetic payload is **not** what the normal scorer emits. Direct backend callers must supply JSON-compatible dictionaries for Redis/PG; InMemory permits a wider set of Python objects. No evidence justifies the hypothesized ordinary-state serialization P1.

This does not certify custom policy objects or caller-injected internal state. The backend schema also omits axis identities, which is a separate, reproduced semantic failure (F3).

## Integration Consistency Matrix

“Self save/load” means the wrapper's own JSON persistence, not ProfileScorer.save/load.

| Scorer method/property | LangChain | MCP | Airflow Score | Airflow Verify |
|---|---|---|---|---|
| score() | Yes, 158/302 | Yes, 131/275 | Yes, 140 | No |
| update() | Yes, 241 | Yes, 179 | No | Yes, 205 |
| accuracy() | No; derives from audit decisions | No; derives from audit decisions | No | Yes, 217 |
| verified_count() | No; derives from audit decisions | No; derives from audit decisions | No | No |
| conservation_status | Uses set_conservation_status and calculated checks | Same | Reads, 147 | Reads, 216 |
| ProfileScorer.save() | No; self.save(), 319 | No; self.save(), 298 | No | No |
| ProfileScorer.load() | No; self.load(), 355 | No; self.load(), 334 | **No** | No |
| get/restore_checkpoint_state | Yes, 335/391 | Yes, 314/369 | No | No |

C3: LangChain and MCP persist `scorer_observability` containing verified_counts, correct_counts, centroid_drift. Their load paths restore it when present. JSON object keys are converted back to integer indices by restore_checkpoint_state, profile_scorer.py:1493–1501.

Airflow verify persists through update only **when backend is configured**. Without a backend it updates a temporary scorer that is discarded after execute returns; there is no file persistence. With a backend, verification reads fresh category state, but scoring never loads it (F2).

C4: LangChain↔MCP share the LearningState JSON schema and validate category/action/factor axes. `tests/test_mcp_integration.py:132 test_state_compat_with_langchain` is covered by the passing suite. Airflow has no corresponding file-loader or interchange test. Passing a state_path into scorer_config would not work because build_profile_scorer has no such keyword. A migration adapter would be needed; the current Airflow persistence interface is a backend instance.

## Dependency Isolation Verification

| Subprocess configuration | Core imports | state star import | Optional state exports |
|---|---|---|---|
| Base only: block prometheus_client, redis, psycopg2, airflow, langchain_core, fastmcp | OK | OK | None |
| Prometheus available; redis/psycopg2/airflow blocked | OK | OK | None |
| Redis available; prometheus/psycopg2/airflow blocked | OK | OK | RedisStateBackend |
| All locally installed dependencies allowed | OK | OK | RedisStateBackend, PostgresStateBackend |

D2 trace: state/__init__.py:9 and 19 attempt imports; ImportError branches retain installation messages without adding missing classes to __all__; successful branches append the names. Thus star import selects only available classes. No connection is opened simply by importing a backend.

D3: gae/__init__.py:102–115 **does eagerly attempt** optional backend and metrics imports, but each is guarded. The proposition “top-level optional import necessarily breaks base install” is disproved by the matrix above. This is a deviation from strict lazy loading, not an observed base-install P1. No integration wrapper is imported at package root.

pyproject.toml:23–25 retains only numpy as base runtime dependency. Extras are dev, langchain, mcp, prometheus, redis, airflow, postgres. An installed Airflow runtime was not proven by importing the module: airflow.py:20–37 supplies a fallback BaseOperator when Airflow is absent.

## Backward Compatibility Assessment

E1: save() at profile_scorer.py:1421 pickles the scorer object, not a schema-versioned JSON envelope. __getstate__ at 1438 copies its dictionary and removes _backend. P1/P4 additions include _verified_count, _correct_count, _centroid_drift and _scorer_id. Backend clients/locks/connections are excluded. Legacy fallback in __setstate__ (1520–1550) supplies backend=None, scorer_id="default", absent phase-related fields, and zero observability dictionaries.

The existing test at tests/test_metrics.py:130 deletes three observability fields from a **current** state dictionary, then calls __setstate__. It passes, demonstrating missing-field defaults; it is not an actual v0.7.27 artifact.

E2: an authentic v0.7.27 wheel/pickle was unavailable in local dist (only 0.7.25 artifacts). No downgrade crash is asserted. Unknown pickle dictionary keys do not inherently cause failure; the old class's restoration implementation and class references determine compatibility. Actual v0.7.27→0.8.0 and reverse cross-version loading remain unverified (F8).

E3: backend load maps category names and ignores categories absent from the new configuration (1398–1405); a newly added category retains initialization. Shape mismatches raise at 834/843. Equal-shaped reordered action axes pass and silently change the meanings of saved centroids (F3). Backend records contain no actions/factors fingerprint.

## AgentEvolver Edge Case Analysis

- Zero verified decisions on both scorers: observed waiting; accuracy is 0.0 with a zero guard.
- Base has data but candidate has zero: candidate count remains below min_decisions; waiting branch at 78. Base count does not bypass that condition.
- Zero categories: constructed ProfileScorer with tensor shape (0,2,1); observed waiting, since min_decisions must be positive. Factory construction of empty categories may reject earlier.
- Different category counts: constructor raises ValueError at 55–56.
- Equal counts but reordered/different names: accepted and compared positionally; reproduced incorrect promotion (F5).
- Normal zero-data accuracy does not produce NaN. If a consumer supplies NaN accuracy, comparisons against zero fail and cannot satisfy ALL/ANY; the weighted sum becomes NaN and also fails. There is no explicit finite-value validation, but no observed NaN-based promotion.
- Negative timeout: observed ValueError. Zero timeout: observed waiting on immediate evaluation because elapsed=0 and condition is strict > (F7).
- Current timeout test at tests/test_agentevolver.py:64–70 subtracts 0.01 from _started_at; passing suite confirms it no longer relies on sleep. Earlier P12 commentary saying the construction timestamp was missing was incorrect: agentevolver.py:67 already records it. Freezegun as the original cause was not demonstrated.
- evaluate() results wrap maps with MappingProxyType at 87–89; history returns a copy at 101. Direct public EvaluationResult construction accepts mutable dictionaries (F6).
- History grows on each evaluate() call at 95 and has no retention limit. This is an existing audit-trail design characteristic; further memory-impact investigation was not expanded after the stop threshold.

## Ranked Findings

### F1 — P1: save() overwrites newer backend state using a freshly fetched version

**A — Locate:** `ProfileScorer._save_all_categories_to_backend(self) -> None`, profile_scorer.py:1381; entry `save(self, path=None) -> Optional[Path]`, 1408.

**B — Read:** Full bodies 1381–1392 and 1408–1422, plus update/retry 954–1014.

**C — Trace:** Scorers S and U share scorer_id/category. Both have local count=10 after U.load(). S.update commits count=11. U.save enters 1415–1416. Line 1386 fetches the new backend version, discarding its state. Line 1390 extracts U's older count=10; line 1391 supplies the version belonging to count=11. CAS succeeds and replaces newer state with older state.

**D — Verdict:** Exact failing pair: `_state, version = self._backend.load_category(self._scorer_id, category_name)` (1386), then `self._extract_category_state(category_index),` (1390). Observed output: **before stale save 11; after stale save 10**, with no conflict.

**E — Classify:** P1 silent loss of verified learning, all three backends. Fix design: retain the version associated with the local snapshot and reject stale saves; do not borrow a newer version. Multi-category partial-save semantics also need an explicit contract.

### F2 — P1: Airflow scoring ignores persisted learning

**A:** `RGIScoreOperator.execute(self, context: dict[str, Any]) -> dict[str, Any]`, airflow.py:127.

**B:** Full execute 127–150; factory 1647 onward; init_from_config 1556–1644; constructor and score 159–377/443–549.

**C:** Backend contains ten verified outcomes moving action y's centroid toward factor .9. At 132, execute builds a fresh neutral scorer with backend configured. Factory returns an instance without calling load(). At 140, score reads local neutral mu; no backend load occurs. Tied scores choose x. Manually calling load() first yields y.

**D:** `scorer = build_profile_scorer(**self.scorer_config)` and `result = scorer.score(factors, category_index)` skip restoration. Observed **fresh: 0 verified, x; loaded: 10 verified, y; Airflow: x**.

**E:** P1 silent wrong recommendations; persists indefinitely across fresh Airflow task executions. Fix: define initialization/restoration semantics and explicitly load backend state before task scoring. Apply the same freshness contract to separately constructed monitoring/comparison scorers.

### F3 — P1: backend state lacks model-axis identity

**A:** `ProfileScorer._apply_category_state(self, category_index: int, state: Dict) -> None`, profile_scorer.py:829.

**B:** Full body 829–893, extractor 806–827, load mapping 1394–1406.

**C:** Save category a with actions [x,y], trained y centroid at index 1. Construct same scorer_id/category/dimensions with actions [y,x]. load maps a to a at 1403. Shape (2,1) passes 834. At 840, index 1 receives the learned y centroid, but new actions[1] is x. score reports x.

**D:** `self.mu[category_index] = centroids` (840) validates shape but not semantic axis identity. Observed **loaded original actions→y; reversed actions→x**.

**E:** P1 silent model corruption. Fix: persist/validate ordered actions and factors or a schema/configuration fingerprint before applying any category. A fresh scorer_id for a changed schema is a workaround.

### F4 — P1: backend write errors leave uncommitted local learning visible

**A:** `ProfileScorer.update(self, f, category_index, action_index, correct, gt_action_index=None, confidence=None) -> CentroidUpdate`, 954.

**B:** Full 954–1014, _update_local 1016–1263, retry helpers 895–913.

**C:** Backend starts empty. At 986, _update_local moves centroid .5→.505 and verified_count 0→1. save_category at 990 raises OSError before storing anything. Except at 996 catches only VersionConflictError. OSError escapes without reaching the cleanup at 1005–1012.

**D:** `except VersionConflictError as exc:` (996) is the sole write-error handler. Observed **local verified=1, mu[0,1]=.505, backend={}** after failed update.

**E:** P1 inconsistent local/durable state after a failed operation. Fix: restore or invalidate local state on non-conflict failure; handle uncertain commit outcomes by authoritative reload before further use. Do not blindly replay an operation whose commit status is unknown.

### F5 — P1: AgentEvolver compares category positions under candidate labels

**A:** `AgentEvolver.__init__(...)`, agentevolver.py:46; `_comparison_values(self) -> tuple[dict[str,float],dict[str,float],dict[str,float]]`, 110.

**B:** Full class 36–134, including construction, category naming, evaluation and comparison branches.

**C:** Base categories [a,b], accuracies [1,0]. Candidate categories [b,a], accuracies [1,0]. Constructor checks only counts at 55. _category_names (103–108) chooses candidate labels. Line 111 labels base index 0 as b, even though it is a. Deltas become zero. Candidate verified=20 and GREEN, so ALL at 123 passes.

**D:** `base = {name: self.base_scorer.accuracy(index) for index, name in self._categories.items()}` (111). Observed **promoted**, maps both reported as {b:1.0,a:0.0}, although true category a regressed 100 percentage points.

**E:** P1 incorrect promotion verdict. Fix: validate equal unique category identities and align by name, or reject mismatched order explicitly. Different category counts are already rejected.

### F6 — P2: directly constructed EvaluationResult is only shallowly frozen

**A:** `EvaluationResult` generated dataclass constructor, agentevolver.py:23–34.

**B:** Read all fields and evaluate's result construction at 85–95.

**C:** Construct with base_accuracy={'a':.8}. Dataclass frozen prevents field rebinding but retains this mutable dict. Assign r.base_accuracy['a']=999; no field rebinding occurs.

**D:** `base_accuracy: Mapping[str, float]` (28) is only an annotation. Observed **{'a': 999}**. Results produced by evaluate itself are protected by MappingProxyType; this limits the blast radius.

**E:** P2 public immutable-value contract gap. Fix: copy and wrap mappings in __post_init__, covering direct callers too.

### F7 — P2: zero timeout is not guaranteed to expire immediately

**A:** `AgentEvolver.evaluate(self) -> EvaluationResult`, agentevolver.py:71.

**B:** Full 71–96 and constructor 46–69.

**C:** timeout_seconds=0 passes nonnegative validation. Immediate evaluate can see the same clock tick as construction, so elapsed=0. Line 77 checks 0>0, false; no candidate decisions leads to waiting.

**D:** `if elapsed > self.timeout_seconds:` (77). Observed **timeout 0 waiting**. A sufficient-data candidate could proceed to comparison during that tick.

**E:** P2 boundary behavior. Fix: define zero as immediate expiry and use >=; add a deterministic equal-clock test.

### F8 — P2: cross-version pickle assurance exceeds the actual test

**A:** `test_observability_counters_survive_pickle_and_old_state()`, tests/test_metrics.py:130; `ProfileScorer.__setstate__(self, state)`, profile_scorer.py:1520.

**B:** Read full test (130–149), save/load/getstate/setstate bodies.

**C:** Test generates current state at 141; removes three keys at 143; invokes current __setstate__ at 145; observes zeros at 147–149. It never invokes old code, never loads an old-produced artifact and does not exercise downgrade.

**D:** `old_state = scorer.__getstate__()` is a synthetic legacy-state test. No historical release artifact was available to establish both directions.

**E:** P2 critical-path test gap, not a demonstrated downgrade crash. Fix: produce fixtures with the actual published 0.7.27 package and test load both ways in isolated environments; document supported direction explicitly.

## Release Recommendation

**HOLD** for F1–F5. Passing unit tests and matching signatures do not establish correct persistence or comparison semantics.

Required follow-up: regression tests for stale save, fresh backend-backed scoring, axis mismatch, failed writes, and category-aligned comparison; narrowly scoped fixes; full suite and typecheck rerun. Add actual PostgreSQL concurrency/transaction testing and a real Airflow task lifecycle check before claiming those integration paths are production-validated. Resolve or document F6–F8.

This report supersedes the earlier P12 “ready” conclusion for the behaviors reproduced here. No source fixes were made.

v0.8.0 sweep complete.
Source files analyzed: 12 primary files.
Dimensions: 6.
P1 findings: 5.
P2 findings: 3.
P3 findings: 0.
0 source files modified; 1 requested report created.
Release recommendation: HOLD — reproduced state loss and incorrect scoring/promotion behavior.

