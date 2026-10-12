# Deep Research — 2026-10-12 to 2026-10-13

Research branch: `research/deep-20261012`, forked from `main` and isolated from production/earlier experiments.

## Artifacts

- [Cumulative hourly research log](2026-10-12_to_2026-10-13.md): evidence classes, equations, literature, causal hypotheses, controlled test proposals, and implementation status.
- [Depth pipeline audit script](../../scripts/audit_depth_pipeline.py): **non-mutating** inspection of Hugging Face Depth Anything float prediction vs 8-bit PIL visualization, CPU/GPU device selection, inference timing. **Committed but not executed on Spark**.

## How to run first diagnostic on Spark

From the repo root in the training environment with dependencies/models available:

```bash
python scripts/audit_depth_pipeline.py --image /path/to/one/changed_view.png --repeat 3 --device auto > outputs/depth_audit_gpu.json
python scripts/audit_depth_pipeline.py --image /path/to/one/changed_view.png --repeat 3 --device cpu > outputs/depth_audit_cpu.json
```

Replace the placeholder image with a real image. `outputs/` is intentionally gitignored. Keep this benchmark separate from the training runs. The GPU/CPU comparison requires the same hardware environment and warmed model.

**Status:** Branch creation and code/notes commits verified via GitHub connector. No Spark/GPU tests have run during this research session; do not cite the audit as measured performance.
