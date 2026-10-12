#!/usr/bin/env python3
"""Non-mutating schedule/exposure audit for CL-Splats continual updates.

Predicts *eligibility* for gsplat DefaultStrategy refinement from the
repository's step convention; does not claim any Gaussian was actually grown.
Also summarizes optional existing diagnostic JSONs. Stdlib only.

Usage:
  python scripts/audit_convergence_schedule.py --views 5 15 30 50 \
    --iterations 100 500 600 601 1000 3000 \
    --diagnostics outputs/diagnostics/diag_5_100.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def eligible_refine_steps(iterations: int, start: int, every: int,
                          stop: int, reset: int, pause: int = 0) -> list[int]:
    """Return 1-based training iterations; gsplat receives 0-based step."""
    return [
        step + 1 for step in range(iterations)
        if step < stop and step > start and step % every == 0
        and step % reset >= pause
    ]


def exposure_distribution(views: int, iterations: int) -> dict:
    """The camera stack visits every view once before refilling."""
    q, r = divmod(iterations, views)
    return {
        "full_cycles": q,
        "partial_cycle_views": r,
        "min_exposure_per_view": q,
        "max_exposure_per_view": q + (r > 0),
        "mean_exposure_per_view": iterations / views,
    }


def lr_at_update(iteration: int, lr_init: float, lr_final: float,
                 max_steps: int) -> float:
    t = min(max(iteration / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def inspect_diag(path: Path, first_growth: int | None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    iters = data.get("iterations")
    delta = data.get("gaussian_count_delta")
    note = None
    if isinstance(iters, int) and isinstance(delta, (int, float)):
        if first_growth is not None and iters < first_growth and delta > 0:
            note = (
                "Positive count delta before default-strategy growth eligibility: "
                "check custom initialization, config, strategy version, or provenance."
            )
        elif first_growth is not None and iters < first_growth and delta < 0:
            note = (
                "Negative delta before growth is possible through CL-specific "
                "active-only hard pruning at prune_every boundaries."
            )
    return {
        "file": str(path),
        "selected_views": data.get("selected_views"),
        "num_views": data.get("num_views"),
        "iterations": iters,
        "gaussians_before_train": data.get("gaussians_before_train"),
        "gaussians_after_train": data.get("gaussians_after_train"),
        "gaussian_count_delta": delta,
        "final_test_psnr": data.get("final_test_psnr"),
        "final_test_ssim": data.get("final_test_ssim"),
        "optimization_sec": data.get("optimization_sec"),
        "note": note,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=int, nargs="+", default=[5, 15, 30, 50])
    p.add_argument("--iterations", type=int, nargs="+",
                   default=[100, 500, 600, 601, 1000, 3000, 5000])
    p.add_argument("--densify-from", type=int, default=500)
    p.add_argument("--densify-until", type=int, default=15000)
    p.add_argument("--densify-every", type=int, default=100)
    p.add_argument("--prune-every", type=int, default=50)
    p.add_argument("--reset-every", type=int, default=2**31 - 1,
                   help="CL phase disables opacity reset using this sentinel")
    p.add_argument("--lr-init", type=float, default=1.6e-4)
    p.add_argument("--lr-final", type=float, default=1.6e-6)
    p.add_argument("--lr-max-steps", type=int, default=30000)
    p.add_argument("--scene-extent", type=float, default=1.0)
    p.add_argument("--diagnostics", type=Path, nargs="*", default=[])
    args = p.parse_args()

    if any(v <= 0 for v in args.views) or any(i < 0 for i in args.iterations):
        p.error("Views must be positive and iterations nonnegative")
    if args.densify_every <= 0 or args.reset_every <= 0 or args.prune_every <= 0:
        p.error("Refinement, reset and prune intervals must be positive")
    max_iter = max(args.iterations)
    all_refines = eligible_refine_steps(
        max(max_iter, args.densify_from + args.densify_every * 3),
        args.densify_from, args.densify_every,
        args.densify_until, args.reset_every,
    )
    first_growth = all_refines[0] if all_refines else None
    cases = []
    for views in args.views:
        for iterations in args.iterations:
            eligible = eligible_refine_steps(
                iterations, args.densify_from, args.densify_every,
                args.densify_until, args.reset_every,
            )
            cases.append({
                "views": views,
                "iterations": iterations,
                **exposure_distribution(views, iterations),
                "eligible_refinement_iterations_1_based": eligible,
                "possible_active_only_prune_iterations_1_based":
                    list(range(args.prune_every, iterations + 1, args.prune_every)),
                "position_lr_final_iteration": (
                    lr_at_update(iterations,
                                 args.lr_init * args.scene_extent,
                                 args.lr_final * args.scene_extent,
                                 args.lr_max_steps) if iterations else None
                ),
            })

    diagnostics = []
    for path in args.diagnostics:
        try:
            diagnostics.append(inspect_diag(path, first_growth))
        except (OSError, ValueError) as exc:
            diagnostics.append({"file": str(path), "error": str(exc)})

    print(json.dumps({
        "status": "schedule prediction, NOT a GPU benchmark",
        "source_assumption": (
            "CL-Splats passes _timestep_iter (0-based) to gsplat DefaultStrategy; "
            "upstream refines when step>refine_start_iter and step%refine_every==0. "
            "Verify the installed gsplat version before causal attribution."
        ),
        "first_default_growth_eligible_iteration_1_based": first_growth,
        "important": [
            "100 vs 500 iterations do NOT straddle default gsplat growth.",
            "Active-only hard pruning may still change count at iterations 50,100,...",
            "Per-step PSNR/SSIM in _train_step are measured before optimizer.step.",
            "All estimates ignore actual masks, gradients, thresholds and hardware."
        ],
        "cases": cases,
        "diagnostics": diagnostics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
