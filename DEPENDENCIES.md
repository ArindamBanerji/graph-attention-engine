# GAE Dependencies

## Consumed by (Tier 1 — breaking changes are production incidents)
- gen-ai-roi-demo-v4-v50/backend/app/domains/soc/ — ProfileScorer,
  kernels, DomainConfig. 290+ call sites. SOC tensor (6,4,6).
- s2p-copilot/backend/app/domains/s2p/ — ProfileScorer, DomainConfig.
  58+ call sites. S2P tensor (5,5,8).
- copilot-sdk — protocol definitions reference GAE types

## Depends on (Tier 1 — do not add new dependencies)
- numpy (core computation)
- scipy (statistical tests in experiments only)
- Nothing else. No database. No network. No event loop.

## Verification after any change
1. python -m pytest tests/ -v (536 tests)
2. If Tier 1 function changed: grep in gen-ai-roi-demo-v4-v50
   AND s2p-copilot
3. If math changed: verify against math_synopsis_v13
