"""Prompt 0 GAE: Structural map of Graph Attention Engine.
Run from graph-attention-engine-v50/ after 'code-review-graph build'.
"""
import sqlite3
import os

db_path = os.path.join(".code-review-graph", "graph.db")
if not os.path.exists(db_path):
    print("ERROR: Run 'code-review-graph build' first")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

KEY_FILES = [
    "profile_scorer.py",
    "calibration.py",
    "convergence.py",
    "kernels.py",
    "kernel_selector.py",
    "covariance.py",
    "evaluation.py",
    "judgment.py",
    "ablation.py",
    "oracle.py",
    "synthetic.py",
    "enrichment_advisor.py",
]

SEPARATOR = "=" * 70

# ── SECTION 1: Function inventory ───────────────────────────────────
print(SEPARATOR)
print("SECTION 1: FUNCTION INVENTORY — GAE SOURCE FILES")
print(SEPARATOR)

for kf in KEY_FILES:
    funcs = conn.execute(
        "SELECT name, kind, line_start, line_end FROM nodes "
        "WHERE file_path LIKE ? AND kind IN ('Function', 'Method', 'Class') "
        "ORDER BY line_start",
        (f"%gae/{kf}",)
    ).fetchall()
    if not funcs:
        # Try alternate path patterns
        funcs = conn.execute(
            "SELECT name, kind, line_start, line_end FROM nodes "
            "WHERE file_path LIKE ? AND kind IN ('Function', 'Method', 'Class') "
            "ORDER BY line_start",
            (f"%gae%{kf}",)
        ).fetchall()
    if not funcs:
        continue
    fp = conn.execute(
        "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE ?",
        (f"%{kf}",)
    ).fetchall()
    print(f"\n{'─' * 50}")
    for f in fp:
        parts = f["file_path"].replace("\\", "/").split("/")
        print(f"FILE: {'/'.join(parts[-3:])}")
    print(f"  Functions/Methods/Classes: {len(funcs)}")
    for fn in funcs:
        size = (fn["line_end"] or 0) - (fn["line_start"] or 0)
        print(f"  L{fn['line_start']:>4}  [{fn['kind']:<8}] {fn['name']} ({size} lines)")

# ── SECTION 2: ProfileScorer public API ─────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 2: PROFILESCORER PUBLIC API")
print(SEPARATOR)

ps_methods = conn.execute(
    "SELECT name, line_start, line_end FROM nodes "
    "WHERE file_path LIKE '%profile_scorer%' "
    "AND kind IN ('Function', 'Method') "
    "AND name NOT LIKE '_%' "
    "ORDER BY line_start"
).fetchall()
print("\nPublic methods (no underscore prefix):")
for m in ps_methods:
    size = (m["line_end"] or 0) - (m["line_start"] or 0)
    print(f"  L{m['line_start']:>4}  {m['name']} ({size} lines)")

# ── SECTION 3: Kernel classes ───────────────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 3: KERNEL CLASSES")
print(SEPARATOR)

kernel_classes = conn.execute(
    "SELECT name, line_start, line_end, file_path FROM nodes "
    "WHERE kind = 'Class' AND (name LIKE '%Kernel%' OR name LIKE '%kernel%') "
    "ORDER BY file_path, line_start"
).fetchall()
for k in kernel_classes:
    parts = k["file_path"].replace("\\", "/").split("/")
    size = (k["line_end"] or 0) - (k["line_start"] or 0)
    print(f"  {'/'.join(parts[-2:])}:L{k['line_start']}  {k['name']} ({size} lines)")

# ── SECTION 4: Cross-file dependencies ──────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 4: WHO IMPORTS WHAT (cross-module)")
print(SEPARATOR)

for kf in KEY_FILES:
    imports = conn.execute(
        "SELECT DISTINCT target_qualified FROM edges "
        "WHERE file_path LIKE ? AND kind = 'IMPORTS_FROM' "
        "ORDER BY target_qualified",
        (f"%gae/{kf}",)
    ).fetchall()
    if not imports:
        imports = conn.execute(
            "SELECT DISTINCT target_qualified FROM edges "
            "WHERE file_path LIKE ? AND kind = 'IMPORTS_FROM' "
            "ORDER BY target_qualified",
            (f"%gae%{kf}",)
        ).fetchall()
    if not imports:
        continue
    # Filter to only gae-internal imports
    internal = [i for i in imports if "gae" in (i["target_qualified"] or "").lower()]
    if internal:
        print(f"\n{kf} imports from GAE:")
        for i in internal:
            print(f"  ← {i['target_qualified']}")

# ── SECTION 5: Key function callers ─────────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 5: KEY FUNCTION CALL SITES")
print(SEPARATOR)

key_fns = [
    "score", "update", "freeze", "unfreeze",
    "compute_eta_override", "compute_n_half", "compute_per_factor_n_half",
    "compute_iks", "recommend", "preliminary_recommendation",
    "record_comparison", "compute_ece", "run_evaluation",
    "compute_judgment", "run_ablation",
]
for fn_name in key_fns:
    callers = conn.execute(
        "SELECT source_qualified, file_path, line FROM edges "
        "WHERE target_qualified LIKE ? AND kind = 'CALLS' "
        "ORDER BY file_path",
        (f"%{fn_name}%",)
    ).fetchall()
    # Deduplicate by file
    if callers:
        files = set()
        print(f"\n{fn_name}() — {len(callers)} call sites:")
        for c in callers[:8]:
            parts = c["file_path"].replace("\\", "/").split("/")
            short = "/".join(parts[-3:])
            src = c["source_qualified"].split("::")[-1]
            print(f"  {short}:L{c['line']} — {src}")
        if len(callers) > 8:
            print(f"  ... and {len(callers) - 8} more")

# ── SECTION 6: scorer.mu check (MUST BE 0) ─────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 6: SAFETY CHECKS")
print(SEPARATOR)

# Check for .mu attribute access
mu_nodes = conn.execute(
    "SELECT file_path, line_start, name FROM nodes "
    "WHERE name = 'mu' AND file_path LIKE '%gae%'"
).fetchall()
print(f"\n'mu' as attribute/variable in GAE: {len(mu_nodes)}")
for m in mu_nodes:
    parts = m["file_path"].replace("\\", "/").split("/")
    print(f"  {'/'.join(parts[-3:])}:L{m['line_start']}")

# Check centroids references
centroid_nodes = conn.execute(
    "SELECT file_path, line_start FROM nodes "
    "WHERE name = 'centroids' AND file_path LIKE '%gae%'"
).fetchall()
print(f"\n'centroids' in GAE: {len(centroid_nodes)}")

# ── SECTION 7: Test file inventory ──────────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 7: TEST FILES")
print(SEPARATOR)

test_files = conn.execute(
    "SELECT DISTINCT file_path FROM nodes "
    "WHERE file_path LIKE '%test%' AND kind IN ('Function', 'Method') "
    "ORDER BY file_path"
).fetchall()

for tf in test_files:
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM nodes "
        "WHERE file_path = ? AND kind = 'Function' AND name LIKE 'test_%'",
        (tf["file_path"],)
    ).fetchone()
    parts = tf["file_path"].replace("\\", "/").split("/")
    short = "/".join(parts[-3:])
    if count["cnt"] > 0:
        print(f"  {count['cnt']:>3} tests  {short}")

# ── SECTION 8: Consumer contract surface ────────────────────────────
print(f"\n{SEPARATOR}")
print("SECTION 8: CONSUMER CONTRACT — what external repos call")
print(SEPARATOR)

# Functions in profile_scorer.py that are called from test_consumer_contracts
consumer_tests = conn.execute(
    "SELECT target_qualified, line FROM edges "
    "WHERE file_path LIKE '%consumer_contract%' AND kind = 'CALLS' "
    "ORDER BY line"
).fetchall()
if consumer_tests:
    print("\nConsumer contract test calls:")
    seen = set()
    for c in consumer_tests:
        target = c["target_qualified"].split("::")[-1] if c["target_qualified"] else "?"
        if target not in seen:
            seen.add(target)
            print(f"  L{c['line']:>4} → {target}")
else:
    print("\n  (no consumer contract test file found in graph)")

# ── SUMMARY ─────────────────────────────────────────────────────────
print(f"\n{SEPARATOR}")
print("GAE STRUCTURAL MAP COMPLETE")
print(SEPARATOR)

conn.close()
