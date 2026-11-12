# Project working instructions

Read `README.md`, `plan.md`, and `docs/RELATED_WORK.md` before changing the research direction. When present in the private workspace, also read `memory/STATE.md`, `memory/PROJECT.md`, and `.local/AGENTS.private.md` for the current handoff and cluster operating rules.

- Use all four BreakHis magnifications and support any nonempty available subset. The original 40X notebooks are historical experiments, not the proposed architecture.
- Keep facts, hypotheses, decisions, and measured results distinct. Do not invent novelty, metrics, completed work, citations, or publication acceptance.
- Do not assume exact correspondence between equally numbered fields at different magnifications. Preserve raw identifiers and audit grouping before selecting a protocol.
- Keep all related images, magnifications, and derived examples in the same split. Validation controls model selection; outer test outcomes do not.
- Measure strong masked mean/gating/attention baselines with the same encoder, folds, seeds, and tuning budget as candidate mechanisms.
- Test data parsing, split disjointness, missing-view invariance, permutation invariance, numerical stability, and save/reload behavior. Use `python3 -m unittest discover -s tests -v` for lightweight checks.
- Save dataset, split, encoder, code, and configuration revisions with every experiment. Never describe synthetic checks as reproduced research performance.
- Use scheduler jobs for heavy work on shared clusters. Inspect the queue and allocation policy before launching bounded arrays; leave unrelated jobs alone.
- Keep datasets, model weights, private reference documents, session memory, credentials, and machine-specific paths out of public Git history.
- Do not change or redistribute the original supplied reference files automatically. Cite their papers; maintain their local provenance.
- Update the plan and measured research status after material work. Private workspaces should also update running memory before a handoff.
