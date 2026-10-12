# Deep Research — 2026-10-12 to 2026-10-13

Research branch: `research/deep-20261012`, forked from `main` and isolated from production/earlier experiments.

## Artifacts

- [Cumulative hourly research log](2026-10-12_to_2026-10-13.md): evidence classes, equations, literature, causal hypotheses, controlled test proposals, and implementation status.
- [Convergence schedule and exposure audit](../../scripts/audit_convergence_schedule.py): **non-mutating, stdlib-only** prediction of the first eligible gsplat growth step (iteration 601 with current defaults), camera exposure counts, possible active-only pruning steps, and optional diagnostics summary. Committed; **not executed on Spark**.
- [Depth pipeline audit script](../../scripts/audit_depth_pipeline.py): **non-mutating** inspection of Hugging Face Depth Anything float prediction vs 8-bit PIL visualization, CPU/GPU device selection, inference timing. **Committed but not executed on Spark**.

## Convergence schedule audit (no GPU required)

From the repository root:

```bash
python scripts/audit_convergence_schedule.py --views 5 15 30 50 --iterations 100 500 600 601 1000 3000 > outputs/convergence_schedule.json
python scripts/audit_convergence_schedule.py --views 5 --iterations 100 500 1000 --diagnostics outputs/diagnostics/diag_5_100.json outputs/diagnostics/diag_5_500.json > outputs/convergence_diag_summary.json
```

The first command predicts *eligible* growth iterations from inspected gsplat logic; it does **not** verify that any Gaussian actually duplicated/split. The optional second command tolerates missing diagnostic files and records errors. Check the installed gsplat version and run on the same experiment branch/config. The 100→500 quality gain cannot be attributed to default growth at 500 because the first eligible growth is the **601st** training iteration with current settings. Active-only pruning can still occur before then.

## How to run first diagnostic on Spark

From the repo root in the training environment with dependencies/models available:

```bash
python scripts/audit_depth_pipeline.py --image /path/to/one/changed_view.png --repeat 3 --device auto > outputs/depth_audit_gpu.json
python scripts/audit_depth_pipeline.py --image /path/to/one/changed_view.png --repeat 3 --device cpu > outputs/depth_audit_cpu.json
```

Replace the placeholder image with a real image. `outputs/` is intentionally gitignored. Keep this benchmark separate from the training runs. The GPU/CPU comparison requires the same hardware environment and warmed model.

**Status:** Branch creation and code/notes commits verified via GitHub connector. No Spark/GPU tests have run during this research session; do not cite the audit as measured performance.
