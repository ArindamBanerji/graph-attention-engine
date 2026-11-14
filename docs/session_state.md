--- R1-PACKAGING-GAE (item 25) ---
Date: 2026-10-16
Prompt: codex_r1_packaging_gae_v1.md
Model: Spark+medium

GAP TABLE:

| Item | Status | Detail |
|---|---|---|
| pyproject.toml | OK | PEP 517 project metadata is present and complete. |
| LICENSE | OK | Apache License 2.0 text is present. |
| README.md | OK | Install, quick-start, API, testing, documentation, and license sections are present. |
| __version__ defined | OK | `gae/__init__.py`: `0.7.25`. |
| version alignment | OK | `pyproject.toml`, `gae.__version__`, installed metadata, and latest `v0.x` package-release tag are `0.7.25`. Internal project milestone tags use a separate `v5.x` stream (latest `v5.80`). |
| build system | OK | setuptools via `setuptools.build_meta`; wheel enabled. |
| dependencies listed | OK | Runtime import audit found only NumPy outside the standard library and local `gae` modules; `numpy>=1.24` is declared. |
| entry_points | OK | Not applicable: GAE is a library and exposes no CLI. |
| classifiers | OK | Seven classifiers cover beta status, audience, Python 3/3.10-3.12, and AI. |
| author/maintainer | OK | Author metadata retained; maintainer added as Dakshineshwari LLC. |

Gaps fixed: added `project.maintainers = [{name = "Dakshineshwari LLC"}]`.
pyproject.toml version: 0.7.25
LICENSE: Apache-2.0
Tests: 1244 passed
Install: clean (`pip install -e . --break-system-packages`); installed distribution and `gae.__version__` both report 0.7.25.
Mypy: `gae/__init__.py` passed with 0 errors.
Self-verification: 5/5 PASS

| CHECK | Description | Result | Evidence |
|---|---|---|---|
| 1 | pyproject.toml fields | PASS | `graph-attention-engine v0.7.25`; all required fields present. |
| 2 | LICENSE exists | PASS | `LICENSE` contains Apache License 2.0. |
| 3 | Version alignment | PASS | TOML `0.7.25` equals latest package-release tag `v0.7.25`; code and installed metadata also report `0.7.25`. |
| 4 | Clean import | PASS | `import gae` resolves this checkout and reports `0.7.25`. |
| 5 | Tests passed | PASS | `1244 passed, 0 failed` in 27.34s. |

Layer 1 self-check: all required files were read before the task edit; checks 1-5 PASS; mypy PASS; changes are limited to packaging metadata and this session log; test count increased from the historical 1,237 baseline to 1,244; no production logic changed.

--- GAE-LANGCHAIN-JUDGMENT-MEMORY ---
Date: 2026-10-31
Model: Spark+high

Implemented an optional LangChain Judgment Memory bridge around `ProfileScorer`.

Files created:
- `gae/integrations/__init__.py`
- `gae/integrations/langchain.py`
- `tests/test_langchain_integration.py`
- `examples/langchain_quickstart.py`

Files modified:
- `gae/__init__.py` — documented the optional LangChain integration without importing it.
- `pyproject.toml` — added `langchain = ["langchain-core>=0.2"]` optional dependency.
- `docs/session_state.md` — recorded this implementation and validation.

Behavior delivered:
- Scores LangChain tool starts as decisions, associates tool outputs, and learns only from explicit `verify()` / `verify_batch()` outcomes.
- Supports custom factor extraction and tool/category mappings, RED conservation handling (`warn`, `raise`, `log`), thread-safe concurrent callbacks, per-category accuracy, Judgment explanations, and Eq. IKS centroid-drift reporting.
- Persists flattened centroid geometry through `gae.store.LearningState` plus decision/counter metadata, using the existing atomic `save_state()` / `load_state()` functions.
- State auto-loads when `state_path` exists and auto-saves after verification.

API diagnosis:
- `save_state()` requires `gae.store.LearningState`, not a dictionary. The root `gae.LearningState` currently names the separate learning-module state type and is not accepted by `gae.store.save_state`; the integration therefore imports the persistence type from `gae.store` explicitly.
- `build_profile_scorer()` cannot infer the factor dimension from an empty centroid mapping even though it receives `n_factors`; the integration supplies a complete neutral 0.5 centroid mapping through the factory.

Validation:
- New tests: 15 passed (`tests/test_langchain_integration.py`), including a real `BaseCallbackHandler` subclass check and FakeListLLM agent loop in the available environment.
- Full suite: 1,259 passed, 0 failed (1,244 existing + 15 new), 13,254 warnings, 9.13s.
- Mypy: all requested files passed with 0 errors.
- Example: 20 decisions, 15 verified, 73.3% accuracy, GREEN conservation, IKS 5.03; Day 1 probabilities `[0.25, 0.25, 0.25, 0.25]` changed to Day 20 `[0.3138, 0.23, 0.2281, 0.2281]`; state reload preserved all 15 verified updates.

--- GAE-MCP-JUDGMENT-MEMORY ---
Date: 2026-10-31
Model: Spark+high

Implemented a FastMCP server and standalone CLI for GAE Judgment Memory.

Files created:
- `gae/integrations/mcp.py`
- `gae/integrations/mcp_cli.py`
- `tests/test_mcp_integration.py`
- `examples/mcp_quickstart.py`
- `examples/mcp_client_config.json`
- `examples/mcp_config.yaml`

Files modified:
- `gae/__init__.py` — documented the optional MCP integration without importing it.
- `pyproject.toml` — added `mcp = ["fastmcp>=2.0"]` and the `rgi-mcp-server` console entry point.
- `docs/session_state.md` — recorded implementation and validation.

Behavior delivered:
- Six MCP tools: `rgi_score`, `rgi_verify`, `rgi_conservation`, `rgi_decisions`, `rgi_accuracy`, and `rgi_explain`.
- Dependency-neutral `JudgmentMemoryService` supports direct use without FastMCP; `create_server()` binds the same methods to FastMCP.
- Exact factor/category validation, thread-safe scoring and verification, conservation monitoring, per-category accuracy, Eq. IKS, Judgment explanations, and automatic state persistence after verification.
- MCP and LangChain use the same `gae.langchain.judgment-memory.v1` `LearningState` format. Tests validate state interchange in both directions.
- CLI merges command-line and YAML configuration and supports stdio plus HTTP/SSE transports.

Validation:
- FastMCP installed and validated at version 2.0.0.
- New tests: 16 passed (`tests/test_mcp_integration.py`).
- Full suite: 1,275 passed, 0 failed (1,259 existing + 16 new), 13,286 warnings, 19.10s.
- Mypy: all four requested files passed with 0 errors.
- Editable install: `graph-attention-engine==0.7.25` installed successfully.
- Stdio smoke test: `rgi-mcp-server --categories a,b,c --actions x,y,z --factors f1,f2,f3` entered the protocol loop and was terminated normally.

Example output summary:
- 20 decisions, 15 verified, 11 correct (73.3%).
- Conservation GREEN: signal 11.0, theta_min 1.5687, headroom 7.012.
- Per-category accuracy: retrieval 80%, generation 80%, routing 60%; IKS 3.84.
- Day 1 probabilities `[0.25, 0.25, 0.25, 0.25]` shifted on Day 20; state reload preserved all 15 verified outcomes.
- Printed responses from all six tools.

CLI `--help` includes:
- `--config`, `--categories`, `--actions`, `--factors`, `--state`
- `--penalty-ratio`, `--q-window`, `--tau`, `--server-name`
- `--stdio | --http`, `--port`

Issues encountered:
- The environment initially had FastMCP 1.0. Two downloads of the latest 4.1.4 release failed with a corrupted HTTP chunk. Installing the minimum declared version, FastMCP 2.0.0, succeeded.
- Pip reported dependency conflicts in unrelated globally installed packages after the FastMCP 2.0.0 upgrade. The GAE editable install and all 1,275 repository tests passed afterward.

--- GAE-MCP-ECOSYSTEM-STATUS-DASHBOARD ---
Date: 2026-10-31
Model: Spark+high

Created MCP directory listing assets and a fault-tolerant system status dashboard.

Files created:
- `mcp-registry/server.json`
- `mcp-registry/README.md`
- `mcp-registry/publish.sh`
- `examples/status_dashboard.py`
- `tests/test_status_dashboard.py`

QuickBooks diagnostic verdict: **PARTIALLY BUILT**.
- Live connector: `../copilot-sdk/apps/purchasing/backend/app/connectors/qbo_connector.py` defines `QBOConnector`, implements the source-connector contract, refreshes OAuth tokens through `intuitlib`, queries vendors/bills/purchase orders/payments, and normalizes data for purchasing workflows.
- Router: `../copilot-sdk/apps/purchasing/backend/app/routers/qbo_router.py` mounts `/api/purchasing/qbo`, selects live or demo mode, and reads `QBO_CLIENT_ID`, `QBO_CLIENT_SECRET`, `QBO_REFRESH_TOKEN`, `QBO_REALM_ID`, and `QBO_SANDBOX`.
- API URLs exist for `https://sandbox-quickbooks.api.intuit.com/v3` and `https://quickbooks.api.intuit.com/v3`.
- Demo/fixture implementation: `../copilot-sdk/apps/purchasing/backend/app/connectors/mock_qbo.py` provides deterministic QBO-shaped data; `../copilot-sdk/apps/purchasing/backend/tests/test_qbo_connector.py` covers connector normalization and router endpoints. Purchasing unit and E2E tests also reference QBO provenance/data.
- UI integration exists in purchasing API/components and E2E specs, including `qbo-supplier.spec.ts`.
- No QuickBooks references were found in the targeted `s2p-copilot` or `ci-platform` Python scans.
- The implementation is not classified COMPLETE because no OAuth authorization/callback flow or live credential-backed integration test was found; deployment expects an existing refresh token and otherwise uses demo data.

Status dashboard behavior:
- Checks the GAE package/version, ProfileScorer, conservation, state persistence, LangChain and MCP integrations, four copilot health endpoints, PostgreSQL+AGE, four public data services, QuickBooks/Toast/Alpaca connector modules, fixture JSON, and three MCP directories.
- Supports colored terminal output, `--json`, and `--copilots-only`; all network checks use a three-second timeout and failures are contained.
- Live run: GAE v0.7.25, ProfileScorer, conservation, state persistence, LangChain, MCP, Trading, Purchasing, DataOps, S2P, and PostgreSQL+AGE reported healthy.
- QuickBooks, Toast POS, and Alpaca connector source files were detected and reported partial because presence does not prove configured live connectivity.
- No JSON fixture files were present in this repository's `fixtures/` directory.
- SAP, Celonis, Open-Meteo, FRED, and MCP directory probes failed cleanly due this Python environment's local CA-chain verification error. JSON summary: 9 OK, 3 partial, 8 unavailable.

Validation:
- New tests: 5 passed (`tests/test_status_dashboard.py`).
- Full suite: 1,280 passed, 0 failed (1,275 existing + 5 new), 13,296 warnings, 7.36s.
- Mypy: `examples/status_dashboard.py` and `tests/test_status_dashboard.py` passed with 0 errors.
- Both `python examples/status_dashboard.py` and `python examples/status_dashboard.py --json` exited 0.

Issues encountered:
- An installed third-party package named `examples` shadowed the repository directory during test collection; the test now loads the dashboard explicitly by file path.
- Windows retained `mcp-registry/publish.sh` as mode 0666 because NTFS does not preserve POSIX execute bits and the local Bash launcher was inaccessible. The script has a Bash shebang and can be made executable with `chmod +x mcp-registry/publish.sh` on Linux/macOS before publishing.

--- GAE-PYPI-SMOKE-V0.7.26 ---
Date: 2026-10-31
Model: Spark+medium

Created:
- `examples/pypi_smoke_test.py`

Behavior:
- Exercises nine isolated public-API checks: installed package metadata, a 3x3x5 ProfileScorer, ten scores, eight verified updates, conservation, Judgment, atomic state save/load, the optional LangChain callback, and the optional MCP server.
- Every check catches and reports its own failure so later checks continue.
- Missing LangChain or FastMCP dependencies are handled as valid optional-dependency outcomes, allowing the script to run after only `pip install graph-attention-engine`.
- Uses only root `gae` exports and the two documented optional integration modules; it does not import an internal GAE implementation module.

Full smoke-test output:
```
=== GAE v0.7.26 PyPI Smoke Test ===
[PASS] Package version: 0.7.26
[PASS] ProfileScorer: 3x3x5
[PASS] Score: action=use, confidence=0.999
[PASS] Update: 8 verified (6 correct, 2 incorrect)
[PASS] Conservation: GREEN
[PASS] Judgment: action=use, confidence_tier=high
[PASS] State save/load: centroids match
[PASS] LangChain callback: available
[PASS] MCP server: available
=== 9/9 PASSED ===
```

Validation:
- `python examples/pypi_smoke_test.py`: exit 0, 9/9 passed.
- Mypy: `examples/pypi_smoke_test.py` passed with 0 errors.
- Full suite: 1,280 passed, 0 failed, 13,296 warnings, 9.24s.

Issue encountered:
- The root symbol `gae.LearningState` is the online-learning state from `gae.learning`, while the public `save_state()` function asserts the distinct persistence state from `gae.store`. That persistence class is not exported at package root. To preserve the public-only constraint, the smoke test creates a schema-valid bootstrap JSON with the current flattened centroids, obtains the persistence object through public `load_state()`, then verifies the public `save_state()`/`load_state()` round trip and centroid equality. This export ambiguity remains a package API issue.
--- EXP-C1b: Extended kernel comparison ---
Date: 2026-11-01
Model: Spark+high

- Kernels tested: dot product, L2 nearest centroid, cosine similarity,
  Mahalanobis (OAS shrinkage), learned positive diagonal metric.
- Design decisions: retained EXP-C1's generator, categories, actions, accuracy
  metric, 10 fixed seeds, and 10,000 alerts/seed; used a deterministic 50/50
  split per seed and reported only the 5,000-alert test half. Mahalanobis fits
  full covariance to training residuals with NumPy-only OAS shrinkage. The
  learned comparator fits six positive per-factor weights by multiclass
  cross-entropy. These two fitted kernels intentionally receive a training-data
  advantage over parameter-free dot/L2/cosine.
- Results (accuracy +/- population std across seeds): dot 60.900% +/- 0.704%;
  L2 97.888% +/- 0.152%; cosine 96.448% +/- 0.366%; Mahalanobis 97.898% +/-
  0.161%; learned metric 97.694% +/- 0.163%.
- Tests: 9 new, 1,289 total, 0 failures. Focused kernel/selector suite: 108
  passed. Mypy: all changed GAE and experiment/chart files passed with 0 errors.
- Key finding: Mahalanobis nominally won by 0.010 pp over L2, an immaterial edge
  within seed variation, while the learned metric trailed L2 by 0.194 pp; full
  covariance defines the boundary but adds no practical value here, so L2
  remains the appropriate default for these bounded low-correlation features.

## P3 StateBackend Protocol — COMPLETE

- Baseline tests: 1,288 passed, 1 pre-existing optional LangChain dependency failure (1,289 collected)
- New tests: 11
- Final total: 1,299 passed, 1 skipped (1,300 collected; zero failures)
- Files created: `gae/state/__init__.py`, `gae/state/protocol.py`, `gae/state/memory_backend.py`, `tests/state/__init__.py`, `tests/state/test_protocol.py`, `docs/state_backend_design.md`
- mypy: clean
- Design decisions: per-category CAS, opaque version tokens, upsert semantics, InMemory reference impl
- Notes: P4 will map a category's centroid/count slice into these records; current local persistence remains unchanged. The optional LangChain agent test now skips when its installed packages are incompatible.

## P6 Review of P3 StateBackend — COMPLETE

- Review result: ISSUES FOUND AND FIXED
- P1 issues: 1 — corrected the P4 design guidance: count-decayed, clipped centroid updates are serialized by CAS retries but are not order-independent weighted means; retries must reapply the verification to freshly loaded state rather than merge stale centroids.
- P2 issues: 3 — conformance tests do not explicitly assert stale initial-token rejection or defensive-copy isolation; Barrier-based concurrency tests rely on the suite timeout rather than local timeouts; empty scorer/category identifiers are accepted as generic dictionary keys, so P4 must validate its identifier policy and category-state schema.
- P3 issues: 0
- Tests after review: 1,305 passed, 1 skipped, 0 failed (full suite); 11 state tests passed
- Ready for P4: YES — P4 must use retry-by-replay semantics and define scorer ID/category-state validation.

## P1 Prometheus Exporter — COMPLETE

- Baseline tests: 812 passed before the pre-existing incompatible LangChain-agent dependency test failed; the test is now explicitly skipped because newer LangChain removes `langchain_core.pydantic_v1`.
- New tests: 6 passed (`tests/test_metrics.py`).
- Final total: 1305 passed, 1 skipped, 0 failed.
- Files created: `gae/metrics.py`, `tests/test_metrics.py`.
- Files modified: `gae/profile_scorer.py`, `pyproject.toml`, `tests/test_langchain_integration.py`.
- mypy: clean (`python -m mypy gae/metrics.py --config-file pyproject.toml`).
- Notes: ProfileScorer extended with per-category `accuracy()`, `verified_count()`, and `centroid_drift()` — required for meaningful metrics export. Counters are updated for every verified `update()` call, including gated/frozen paths (with zero drift), and survive checkpoint state. `build_profile_scorer()` now preserves category names so metric labels are meaningful. `prometheus_client>=0.20` is optional under the `prometheus` extra; the base install remains dependency-neutral.

## P2 Review of P1 Prometheus + Scorer APIs — COMPLETE

- Review result: ISSUES FOUND AND FIXED
- P1 issues: 1 — LangChain and MCP public save/load adapters did not persist the new observability counters; both now store and restore the compact counter/drift state, while older state files without it retain zero defaults.
- P2 issues: 2 — `RGIMetrics.start()` delegates repeated-same-port handling to `prometheus_client` rather than being idempotent; `mypy` reports six pre-existing unrelated `ProfileScorer` errors outside the new observability methods (metrics module is clean).
- P3 issues: 0
- Tests after review: 1,308 passed, 1 skipped, 0 failed (full suite); focused persistence/metrics tests: 47 passed, 1 skipped.
- Backward compat verified: YES — legacy pickle state without the three new fields restores zeroed counters; old adapter state files omit optional metadata and retain constructor defaults.
- Ready for v0.7.28 tag: YES — with the two non-blocking notes above.

## P4 StateBackend Scorer Integration — COMPLETE

- Baseline tests: 1,308 passed, 1 skipped
- New tests: 9 (`tests/test_scorer_backend.py`)
- Final total: 1,329 passed, 1 skipped, 0 failed
- Retry strategy: reload-and-replay, max 3 retries after the initial CAS attempt; retries restore runtime counters before replay and never merge stale centroid deltas.
- Category state schema: `centroids`, `counts`, `verified_count`, `correct_count`, `centroid_drift`, plus `category_state` (`phase`, `n_decisions`, `freeze_point`), `dk_weights`, and `cov_inv` when configured.
- Backward compat: pickle load of pre-P4 state tested YES; missing backend state leaves the scorer's initial category state intact.
- Files modified: `gae/profile_scorer.py`
- Files created: `tests/test_scorer_backend.py`
- mypy: clean (`gae/profile_scorer.py`)

## AGENTEVOLVER EXISTING CODE SURVEY

### 1. Definitions

- **SDK orchestration:** `copilot-sdk/copilot_sdk/evolution/evolver.py:27` defines `class AgentEvolver`. Its constructor is `AgentEvolver(ledger: EvolutionLedger | None = None, shadow_runner: ShadowRunner | None = None, promotion_gate: PromotionGate | None = None, plateau_config: PlateauConfig | None = None)`. Its primary operation is `evolve(rule_name: str, decisions: list[dict[str, Any]], conservation_state: Any = None, seed: Any | None = None, decision_id: str | None = None) -> dict[str, Any]`.
- **Portable AE components:** `copilot_sdk/ae/` has no class named `AgentEvolver`. It supplies `DomainEvolutionStrategy` (`strategy.py`), `VariantGenerator.generate(...)`, `FitnessEvaluator.evaluate(...)`, paired-bootstrap `PromotionGate.evaluate(...)`, and GraphStore-backed `EvolutionStore`.
- **SOC adapter:** `gen-ai-roi-demo-v4-v50/backend/app/services/evolver.py:31` defines `DomainEvolutionStrategy`, an adapter around SDK `PromptVariantEvolver`, not a second generic `AgentEvolver`. Its public functions include `get_prompt_variant(...)`, `record_decision_outcome(...)`, `check_for_promotion(alert_type: str)`, `get_evolution_summary(...)`, and `reset_evolver_state()`.
- **Trading adapter:** `copilot-sdk/apps/trading/backend/app/services/trading_evolver.py:161` defines `TradingAgentEvolver`; it owns an SDK `AgentEvolver` plus `DefaultShadowRunner` and `DefaultPromotionGate`.

### 2. References and wiring

- `copilot_sdk/scoring/scorer.py:_setup_evolution()` imports `AgentEvolver`, builds an `InMemoryEvolutionLedger` backed by the configured GraphStore/outbox, registers threshold/factor-weight/action-bias rules, and `_run_evolution()` calls `evolve()` only after at least 10 verified decisions.
- `copilot_sdk/backend/evolution_router.py` constructs and caches either `AgentEvolver` or `PromptVariantEvolver` for FastAPI evolution endpoints.
- SOC `app/routers/evolution.py` exposes history, summaries, rejection reporting, and a legacy process endpoint; SOC `app/routers/admin.py` exposes `/admin/evolution-scan`, `/admin/shadow-start`, and `/admin/promote-evaluate`.
- SOC `app/routers/triage.py:1548` invokes `maybe_shadow_compare()` only for MVP-testable artifacts; no AgentEvolver class is imported there.
- `gae/metrics.py` exposes `rgi_agentevolver_phase{variant}` only as a manually set placeholder; it is not connected to an AgentEvolver instance.
- `ci-platform/` has no `AgentEvolver` definition or direct reference. It provides GraphStore/evolution-event persistence used by SDK ledgers.

### 3. Functional behavior

- `AgentEvolver.evolve()` records `variant_generated`, `shadow_started`, `shadow_completed`, then asks `PromotionGate.evaluate()`. It swaps the active rule only when `gate_result["promoted"]`; otherwise it records `rejected`. `reset()` clears the ledger only.
- `DefaultShadowRunner.run_shadow(variant, decisions, baseline=None)` evaluates candidate and baseline actions over supplied verified decisions and returns total/correct/baseline_correct/accuracy/batch_accuracies/errors.
- `DefaultPromotionGate.evaluate(...)` requires sufficient data, three batches, one-sided two-proportion p-value `< 0.05`, superiority `> 3.0pp`, accuracy `>= 0.70`, variance `<= 0.10`, and conservation promotion permission. Default required shadow size is 1,000.
- SOC AE-01/02/03 is functional: `VariantGenerator.scan(...)` creates candidate `VariantRecord`s; `maybe_shadow_compare(...)` buffers non-production comparisons; batches are persisted as evolution events; `evaluate_promotion(...)` and `execute_promotion(...)` transition a qualifying shadow variant to active; `check_rollback(...)` can roll back a recent active variant.

### 4. Placeholders/stubs

- `gae.metrics.RGIMetrics.set_agentevolver_phase(variant, phase="Shadow")` is a documented manual placeholder, not lifecycle integration.
- `backend/app/services/shadow_mode.py` is explicitly a backwards-compatible re-export stub for `app.framework.shadow_mode`.
- `copilot_sdk/rl/__init__.py` exports rewards, credit assignment, exploration, temporal credit, and budgets; it contains no AgentEvolver symbol or promotion/shadow orchestration.
- Historical `gen-ai-roi-demo-v2/v3/v3.2/v4/v4-v45` and diagnostic files contain older duplicated/canned evolution references. They are not the current `v4-v50` implementation.

### 5. State and data structures

- SDK generic state: active-rule map, plateau cooldown/event-count maps, warning list; `EvolutionEvent(event_type, rule_name, variant_id, timestamp, metadata)`; in-memory append-only `EvolutionLedger`.
- Prompt stack: `VariantSpec`, `VariantStats(successes,total,failures)`, and `CategoryVariantStats`; statuses are `active`, `shadow`, `promoted`, `retired`.
- SOC stack: `VariantRecord(variant_id, artifact_type, category, trigger_key, config, status, created_at, promoted_at, ...)`; statuses are `candidate`, `shadow`, `active`, `rejected`, `rolled_back`; `ShadowComparison` holds production/variant actions, confidence, category, and eventual correct action.
- SOC RL state is separate: reward ledger entries, Thompson posterior alpha/beta matrices, and chain-credit records. It does not change AgentEvolver state.

### 6. Interfaces/dependencies

- Generic AE depends on `EvolutionRule.generate_variant`, `ShadowRunner.run_shadow`, `PromotionGate.evaluate`, `EvolutionLedger.append/get_events`, and the canonical conservation contract `evaluate_conservation_safety`.
- Portable AE depends on `DomainEvolutionStrategy.generate_variants/evaluate_fitness/domain_constraints` and `GraphStore.save_evolution/get_evolution/list_evolutions/save_ledger/save_promotion`.
- SOC shadow/promotion depends on `graph_client.run_query`, `gae.evolution.record_evolution_event`, variant registry state, learning-health/conservation values, and triage outcome filling. It does not call the GAE ProfileScorer directly in the shadow runner.

### 7. Tests

- Generic SDK coverage: `copilot-sdk/tests/evolution/test_evolver.py`, `test_plateau.py`, `test_cross_loop_conservation.py`, `test_prompt_evolver_basic.py`, `test_prompt_promotion_gate.py`, `test_prompt_category_ucb.py`, and `tests/test_ae_framework.py`.
- SOC coverage: `gen-ai-roi-demo-v4-v50/backend/tests/test_variant_generator.py`, `test_shadow_runner.py`, `test_shadow_promotion.py`, `test_promotion_gate.py`, `test_evolution_ledger.py`, and `test_ae_integration.py`.
- Trading coverage: `copilot-sdk/apps/trading/backend/tests/test_trading_evolver.py`, `test_regime_throttle.py`, `test_live_rejection_provenance.py`, and `test_rejection_summary.py`.

### 8. Key configuration/thresholds

- SDK plateau defaults: `plateau_window=10`, `min_improvement_rate=0.2`, `plateau_cooldown=50`; default shadow runner minimum is 10 but default generic gate requires 1,000 decisions.
- Portable paired-bootstrap gate: `min_n=30`, `fpr_threshold=0.05`, `bootstrap_samples=1000`.
- SOC promotion gate: `DELTA_MIN=0.05`, `Q_FLOOR=0.80`, `SIGMA_MAX=0.10`, `MIN_SHADOW_SAMPLES=50`, `MIN_PAIRED_SAMPLES=30`, `BOOTSTRAP_FPR_THRESHOLD=0.05`, `BOOTSTRAP_RESAMPLES=2000`, `MIN_SHADOW_BATCHES=3`, `ROLLBACK_MIN_HOURS=48`, and `ROLLBACK_MIN_DECISIONS=100`.
- SOC explicit ShadowDecision-to-Observation promotion is stricter and manual: confidence `>=0.5`, at least 10 samples, analyst outcome populated, non-synthetic source, non-paused environment, approval token, and actor.

### Other-repository supplement

- **S2P is a second functional implementation.** `s2p-copilot/backend/app/services/s2p_evolver.py` is a GraphStore-backed wrapper around SDK `PromptVariantEvolver`, not the generic `AgentEvolver` class. Exact public entry points are `set_graph_store(graph_store: GraphStore)`, `set_conservation_provider(provider: ConservationStateProvider)`, `get_active_variant(category: str | None = None, family: str | None = None)`, `record_triage_outcome(...)`, `check_promotion(conservation_state: Any = None)`, `propose_variant(dimension_name: str)`, and `shadow_test_variant(variant: dict[str, Any], decisions: list[dict[str, Any]])`.
- **S2P has six explicit tunable dimensions** in `s2p_evolution_dimensions.py`: two auto-approval thresholds, supplier-trust weight, price-variance factor importance, escalation amount threshold, and batch-processing threshold. Each requires three shadow batches; per-batch size is 20, 25, or 30. `S2PEvolutionService` under `app/domains/s2p/evolution/service.py` is separately functional but fixture-backed: it uses `S2PShadowRunner`, `ContextAwareSelector`, and `AutonomousPromotionGate(min_shadow_batches=3, min_win_rate=0.7)`.
- **Legacy demo implementations exist** in `gen-ai-roi-demo-v2`, `v3`, `v3.2`, `v4`, and `v4-v45`. Their `backend/app/services/evolver.py` files use in-memory `PROMPT_STATS`, `ACTIVE_PROMPTS`, `RECENT_PROMOTIONS`, and `WEIGHT_HISTORY`. `check_for_promotion(alert_type: str)` promotes a family candidate only after at least 10 samples and more than 5% observed success-rate improvement. They are functional as demo modules but do not provide the modern graph-backed, paired-shadow, conservation-gated workflow.
- **Blog artifact only:** `blogs/artifacts/s2p_golden_thread/golden_thread.py:282` provides `agent_evolver_check(conservation, category, recent_accuracy, baseline_accuracy) -> Optional[AEThreshold]`; its docstring explicitly calls it simplified and describes production shadow testing/rollback as future behavior.
- **Diagnostics-only references:** `diagnostics/src/pd_feature_map*.py` and `narrative_readiness_collector_*.py` label AgentEvolver as a feature/presentation claim; they contain no execution path.
- No third independent generic `AgentEvolver` class was found outside `copilot_sdk.evolution.evolver.AgentEvolver`; other matches are adapters, legacy demo trackers, test code, documentation, or UI/diagnostic metadata.

## P7 AgentEvolver Design + Implementation — COMPLETE

- This is the scorer-level comparison primitive, **not** a replacement for `copilot_sdk.evolution.evolver.AgentEvolver`.
- Baseline tests: 1,329 passed, 1 skipped
- New tests: 11 (`tests/test_agentevolver.py`)
- Final total: 1,340 passed, 1 skipped, 0 failed
- Files created: `gae/agentevolver.py`, `tests/test_agentevolver.py`
- Files modified: `gae/__init__.py`
- mypy: clean (`gae/agentevolver.py`)
- Comparison modes: `ALL_CATEGORIES`, `WEIGHTED_AVERAGE`, `ANY_IMPROVEMENT`
- Key design: no globals, no domain imports, no deployment actions, conservation-gated read. `EvaluationResult` is frozen and wraps accuracy maps in immutable proxies for an append-only audit trail.

## P5 Redis StateBackend — COMPLETE

- Baseline tests: 1,307 passed, 1 skipped, 1 pre-existing LangChain persistence failure
- New tests: 12
- Final total: 1,319 passed, 1 skipped, 1 pre-existing failure
- Files created: `gae/state/redis_backend.py`, `tests/state/test_redis.py`
- Files modified: `gae/state/__init__.py`, `pyproject.toml`
- mypy: clean
- fakeredis tests: 12 passed
- Notes: Redis hashes store one JSON record per category. WATCH observes the hash key, so unrelated category changes retry internally while target-category changes raise `VersionConflictError`. `KEYS` is used for `list_scorers`; use SCAN for large production keyspaces. The unrelated `tests/test_langchain_integration.py::test_save_load` counter-restore regression remained present before and after this work.

## P11 Airflow Operators — COMPLETE

- Baseline tests: 1,340 passed, 1 skipped
- New tests: 7
- Final total: 1,358 passed, 1 skipped, 0 failed
- Files created: `gae/integrations/airflow.py`, `tests/test_airflow_integration.py`
- Files modified: `pyproject.toml`
- mypy: clean
- Operators: `RGIScoreOperator`, `RGIVerifyOperator`
- Key design: per-execution scorer construction, no module-global scorer state, dynamic callable-path resolution, and XCom-compatible decision/outcome records. The verify payload must retain the scored factor vector and action index so Eq. 4b can be replayed correctly.

## P10 PostgreSQL StateBackend — COMPLETE

- Baseline tests: 1,340 passed, 1 skipped
- New tests: 11 (`tests/state/test_pg.py`)
- Final total: 1,358 passed, 1 skipped
- Files created: `gae/state/pg_backend.py`, `tests/state/test_pg.py`
- Files modified: `gae/state/__init__.py`, `pyproject.toml`
- mypy: clean (`gae/state/pg_backend.py`)
- Test approach: deterministic in-memory DB-API fake that executes the backend's SQL/CAS path; no live PostgreSQL service is required in CI.
- Notes: category writes use `UPDATE ... WHERE version = %s RETURNING version`; a missing initial-version row is inserted, while a stale version raises `VersionConflictError`. The `postgres` extra supplies `psycopg2-binary>=2.9` without changing the base install.

## P12 Final Review + v0.8.0 — COMPLETE

- Review result: PASS WITH NOTES
- P1 issues found and fixed: 2 — `AgentEvolver`'s sleep-based timeout test was flaky under the full pytest/freezegun environment and now advances its monotonic anchor deterministically; `gae` now re-exports the documented StateBackend core types plus installed optional backends and metrics, while absent Redis/PostgreSQL extras provide actionable installation guidance.
- P2 notes: 1 — the requested compatibility smoke command used `ProfileScorer.update(factor_vector=...)`; the established public parameter is `f=`, which was verified to preserve v0.7.27 local scorer and pickle behavior. This is a documentation/example mismatch, not a behavioral regression.
- Final test count: 1,358 passed, 1 skipped
- mypy: clean (12 release-relevant source files)
- Version: 0.8.0
- New public API surface:
  - `gae.AgentEvolver`, `ComparisonMode`, `EvaluationResult`
  - `gae.StateBackend`, `VersionConflictError`, `InMemoryStateBackend`
  - `gae.state.RedisStateBackend` (optional: `redis`)
  - `gae.state.PostgresStateBackend` (optional: `postgres`)
  - `gae.RGIMetrics` / `gae.metrics.RGIMetrics` (optional: `prometheus`)
  - `gae.integrations.airflow.RGIScoreOperator`, `RGIVerifyOperator` (optional: `airflow`)
  - `ProfileScorer`: `backend=`, `scorer_id=`, `accuracy()`, `verified_count()`, `centroid_drift()`
- Ready for git tag + PyPI publish: YES

## SWEEP FIX — F1-F5 P1s + F6-F7 P2s RESOLVED

- Date: 2026-11-14
- F1: Version tracking per category; stale saves conflict, adopt the authoritative category, and retry (initial attempt plus at most three retries). Successful loads/writes retain the corresponding version; pickle copies exclude backend version tokens.
- F2: Auto-load from backend on construction, including direct ProfileScorer construction and both Airflow operators through the factory. Optional load_backend=False defers restoration.
- F3: Ordered action/factor schema metadata in each state dict; axis mismatch raises ValueError before centroid assignment. Factor names now pass through the factory. Unnamed factors use positional names. Category names are recorded, while category restoration remains name-based. Missing legacy schemas remain accepted.
- F4: Snapshot/restore of category state and runtime counters/buffer on non-CAS write failure; conflict attempts also discard their local mutations before reloading and replaying.
- F5: Category alignment by name, including candidate count weights; mismatched or duplicate names rejected.
- F6: EvaluationResult copies and wraps all three accuracy/delta maps in MappingProxyType in __post_init__.
- F7: Timeout uses >= instead of >; exact equal-clock boundary covered.
- Baseline: 1,358 passed, 1 skipped
- New regression tests: 14 in tests/test_sweep_fixes.py
- Focused sweep/AgentEvolver/backend/Airflow suites: 41 passed
- Final total: 1,372 passed, 1 skipped, 0 failed
- mypy: clean for gae/profile_scorer.py, gae/agentevolver.py, gae/integrations/airflow.py, and tests/test_sweep_fixes.py
- All sweep reproduction scenarios F1–F5 pass: YES (separate read-only replay also executed).
- Files modified: gae/profile_scorer.py, gae/agentevolver.py, gae/integrations/airflow.py; added tests/test_sweep_fixes.py.
- Ready for v0.8.0 tag: YES for the requested F1–F7 gates; no tag or publication performed.
- Remaining audit limitation: F8 actual cross-version pickle verification is still outstanding; no live PostgreSQL transaction or real Airflow deployment certification is claimed. A network error after an unknown commit outcome is re-raised, not automatically replayed; callers must reconcile before resubmitting.

## R1 + F8 FIXES — COMPLETE

- Date: 2026-11-14
- R1: `factors=self.factors` passed in LangChain and MCP wrappers; regression tests check both scorer factor names and extracted `_schema` metadata.
- F8: `tests/fixtures/v0.7.27_scorer_state.pkl` created using the requested simulated legacy-state recipe; missing-field restoration, zeroed counters, non-trivial centroids, scoring, and subsequent updates tested. Fixture provenance and recipe documented in `tests/fixtures/README.md`.
- Current v0.8.0 pickle round trip preserves counts, accuracy, decision count, drift, factor names, and centroids; uses pytest `tmp_path` for Windows portability.
- Baseline: 1,372 passed, 1 skipped, 0 failed.
- New tests: 4; focused verification: 4 passed.
- Final total: 1,376 passed, 1 skipped, 0 failed.
- mypy: clean for `gae/integrations/langchain.py` and `gae/integrations/mcp.py` (2 source files).
- Production changes: only the two wrapper factory arguments. No ProfileScorer implementation or persistence logic changed.
- Follow-up backlog: requested R1 and simulated-fixture work complete. The original F8 historical-artifact assurance remains unverified: this fixture was generated with current code, not an installed v0.7.27 release; downgrade compatibility is also not claimed.
- Ready for tag: YES under the existing TAG-with-documented-limitations recommendation; no tag or publication performed.
