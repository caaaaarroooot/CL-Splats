"""Changed-region evaluation for saved CL-Splats PLY checkpoints.

This module evaluates already-trained PLY files without re-running optimization.
For Blender change datasets it builds a fixed ground-truth change mask by
comparing the same camera at t0 and t1, then reports metrics separately inside
and outside the changed region.

Example::

    python -m clsplats.eval_region \
        --ply-dir outputs/ply \
        --data-path data/Blender-Levels/Level-1 \
        --change-type add \
        --timestep 1 \
        --white-background

Expected PLY names are ``ply_<num_views>_<iterations>.ply``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import typer
from loguru import logger
from PIL import Image as PILImage

from clsplats.config import CLSplatsConfig
from clsplats.eval import _load_gaussians_from_ply, _load_scene
from clsplats.trainer import CLSplatsTrainer, _load_gt_image

app = typer.Typer(pretty_exceptions_show_locals=False)

_PLY_RE = re.compile(r"^ply_(\d+)_(\d+)\.ply$")


def _ply_sort_key(path: Path) -> tuple[int, int]:
    match = _PLY_RE.match(path.name)
    if match is None:
        return (10**9, 10**9)
    return int(match.group(1)), int(match.group(2))


def _build_blender_base_gt_index(data_path: str) -> dict[str, Path]:
    """Map image stem -> t0 GT path from base Blender transform files."""
    root = Path(data_path)
    index: dict[str, Path] = {}

    for transforms_name in ("transforms_train.json", "transforms_test.json"):
        transforms_path = root / transforms_name
        if not transforms_path.is_file():
            continue

        with open(transforms_path, "r", encoding="utf-8") as f:
            contents = json.load(f)

        for frame in contents.get("frames", []):
            file_path = Path(frame["file_path"])
            if file_path.suffix == "":
                file_path = file_path.with_suffix(".png")
            full_path = root / file_path
            index.setdefault(full_path.stem, full_path)

    return index


def _image_to_tensor(image: PILImage.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _ssim_map(
    pred_hwc: torch.Tensor,
    gt_hwc: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return a per-pixel SSIM map averaged across RGB channels."""
    pred = pred_hwc.permute(2, 0, 1).unsqueeze(0)
    gt = gt_hwc.permute(2, 0, 1).unsqueeze(0)

    channels = pred.shape[1]
    k1, k2 = 0.01, 0.03
    c1, c2 = k1**2, k2**2

    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    kernel = gaussian.outer(gaussian)[None, None].repeat(channels, 1, 1, 1)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(gt, kernel, padding=pad, groups=channels)
    sigma_xx = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x**2
    sigma_yy = F.conv2d(gt * gt, kernel, padding=pad, groups=channels) - mu_y**2
    sigma_xy = F.conv2d(pred * gt, kernel, padding=pad, groups=channels) - mu_x * mu_y

    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim /= (mu_x**2 + mu_y**2 + c1) * (sigma_xx + sigma_yy + c2)
    return ssim.mean(dim=1).squeeze(0)


def _masked_error_stats(
    pred_hwc: torch.Tensor,
    gt_hwc: torch.Tensor,
    mask_hw: torch.Tensor,
    ssim_hw: torch.Tensor,
) -> dict[str, float | int | None]:
    """Compute RGB error metrics over pixels selected by *mask_hw*."""
    mask = mask_hw.bool()
    pixel_count = int(mask.sum().item())

    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "psnr": None,
            "ssim": None,
            "mse": None,
            "mae": None,
        }

    error = pred_hwc[mask] - gt_hwc[mask]  # [N, 3]
    mse = torch.mean(error**2)
    mae = torch.mean(torch.abs(error))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
    ssim = torch.mean(ssim_hw[mask])

    return {
        "pixel_count": pixel_count,
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "mse": float(mse.item()),
        "mae": float(mae.item()),
    }


def _mean_metric(
    per_view: list[dict[str, object]],
    region: str,
    metric: str,
) -> Optional[float]:
    values: list[float] = []
    for record in per_view:
        region_stats = record.get(region)
        if not isinstance(region_stats, dict):
            continue
        value = region_stats.get(metric)
        if value is not None:
            values.append(float(value))
    return float(sum(values) / len(values)) if values else None


def _summarize_region(
    per_view: list[dict[str, object]],
    region: str,
) -> dict[str, float | int | None]:
    pixel_count = 0
    for record in per_view:
        region_stats = record.get(region)
        if isinstance(region_stats, dict):
            pixel_count += int(region_stats.get("pixel_count", 0))

    return {
        "psnr": _mean_metric(per_view, region, "psnr"),
        "ssim": _mean_metric(per_view, region, "ssim"),
        "mse": _mean_metric(per_view, region, "mse"),
        "mae": _mean_metric(per_view, region, "mae"),
        "pixel_count_total": pixel_count,
    }


def _evaluate_regions(
    *,
    test_cameras,
    render_dir: Path,
    data_path: str,
    white_background: bool,
    device: torch.device,
    mask_threshold: float,
    mask_dilate_px: int,
) -> dict[str, object]:
    """Evaluate changed and unchanged regions using fixed t0-vs-t1 GT masks."""
    base_index = _build_blender_base_gt_index(data_path)
    per_view: list[dict[str, object]] = []

    for cam_info in test_cameras:
        base_path = base_index.get(cam_info.image_name)
        if base_path is None or not base_path.is_file():
            logger.warning("No matching t0 GT for test view '{}'; skipping region metrics.", cam_info.image_name)
            continue

        render_path = render_dir / f"{cam_info.image_name}_render.png"
        if not render_path.is_file():
            logger.warning("Render missing for test view '{}'; skipping region metrics.", cam_info.image_name)
            continue

        base_gt = _load_gt_image(
            str(base_path),
            white_background=white_background,
            composite_alpha=True,
        ).convert("RGB")
        changed_gt = _load_gt_image(
            cam_info.image_path,
            white_background=white_background,
            composite_alpha=True,
        ).convert("RGB")
        rendered = PILImage.open(render_path).convert("RGB")

        if base_gt.size != changed_gt.size or rendered.size != changed_gt.size:
            logger.warning(
                "Resolution mismatch for '{}': t0={}, t1={}, render={}; skipping.",
                cam_info.image_name,
                base_gt.size,
                changed_gt.size,
                rendered.size,
            )
            continue

        base_t = _image_to_tensor(base_gt, device)
        gt_t = _image_to_tensor(changed_gt, device)
        pred_t = _image_to_tensor(rendered, device)

        # Fixed evaluation mask from the actual GT scene change, independent
        # of DINO/change-detection output. A pixel is changed when any RGB
        # channel differs by more than the normalized threshold.
        diff_hw = torch.max(torch.abs(gt_t - base_t), dim=-1).values
        changed_mask = diff_hw > mask_threshold

        if mask_dilate_px > 0:
            kernel = 2 * mask_dilate_px + 1
            changed_mask = (
                F.max_pool2d(
                    changed_mask.float()[None, None],
                    kernel_size=kernel,
                    stride=1,
                    padding=mask_dilate_px,
                )[0, 0]
                > 0
            )

        unchanged_mask = ~changed_mask
        ssim_hw = _ssim_map(pred_t, gt_t)

        changed_stats = _masked_error_stats(pred_t, gt_t, changed_mask, ssim_hw)
        unchanged_stats = _masked_error_stats(pred_t, gt_t, unchanged_mask, ssim_hw)

        total_pixels = int(changed_mask.numel())
        changed_pixels = int(changed_mask.sum().item())
        changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0

        mask_np = (changed_mask.detach().cpu().numpy().astype(np.uint8) * 255)
        PILImage.fromarray(mask_np, mode="L").save(
            render_dir / f"{cam_info.image_name}_change_mask.png"
        )

        record: dict[str, object] = {
            "image_name": cam_info.image_name,
            "base_gt_path": str(base_path),
            "changed_gt_path": cam_info.image_path,
            "changed_pixel_ratio": changed_ratio,
            "changed_region": changed_stats,
            "unchanged_region": unchanged_stats,
        }
        per_view.append(record)

        logger.info(
            "[region eval] {name}: changed={ratio:.3f}  changed PSNR={cpsnr:.2f}  unchanged PSNR={upsnr:.2f}",
            name=cam_info.image_name,
            ratio=changed_ratio,
            cpsnr=(changed_stats["psnr"] if changed_stats["psnr"] is not None else float("nan")),
            upsnr=(unchanged_stats["psnr"] if unchanged_stats["psnr"] is not None else float("nan")),
        )

    changed_ratios = [float(record["changed_pixel_ratio"]) for record in per_view]

    return {
        "mask_source": "same-camera t0 GT vs t1 GT RGB difference",
        "mask_threshold": mask_threshold,
        "mask_dilate_px": mask_dilate_px,
        "region_eval_views": len(per_view),
        "changed_pixel_ratio_mean": (
            float(sum(changed_ratios) / len(changed_ratios)) if changed_ratios else None
        ),
        "changed_region": _summarize_region(per_view, "changed_region"),
        "unchanged_region": _summarize_region(per_view, "unchanged_region"),
        "per_view": per_view,
    }


def _evaluate_one_ply(
    *,
    ply_path: Path,
    scene,
    data_path: str,
    timestep: int,
    output_root: Path,
    metrics_dir: Path,
    white_background: bool,
    sh_degree: int,
    mask_threshold: float,
    mask_dilate_px: int,
) -> dict[str, object]:
    match = _PLY_RE.match(ply_path.name)
    if match is None:
        raise ValueError(f"Unexpected PLY name: {ply_path.name}")

    num_views = int(match.group(1))
    iterations = int(match.group(2))
    run_name = f"{num_views}_{iterations}"
    run_out = output_root / run_name

    cfg = CLSplatsConfig()
    cfg.model.sh_degree = sh_degree
    cfg.white_background = white_background

    logger.info("Loading {}", ply_path)
    gaussians = _load_gaussians_from_ply(str(ply_path), cfg)
    gaussians.active_sh_degree = gaussians.max_sh_degree

    trainer = CLSplatsTrainer.__new__(CLSplatsTrainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.timestep = timestep
    trainer.gaussians = gaussians
    trainer._is_nerf_synthetic = scene.is_nerf_synthetic

    whole_metrics = trainer.evaluate(
        test_cameras=scene.test_cameras,
        timestep=timestep,
        out_dir=run_out,
    )

    render_dir = run_out / f"t{timestep:04d}"
    region_metrics = _evaluate_regions(
        test_cameras=scene.test_cameras,
        render_dir=render_dir,
        data_path=data_path,
        white_background=white_background,
        device=trainer.device,
        mask_threshold=mask_threshold,
        mask_dilate_px=mask_dilate_px,
    )

    result: dict[str, object] = {
        "ply": str(ply_path),
        "num_views": num_views,
        "iterations": iterations,
        "whole_image": whole_metrics,
        **region_metrics,
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"region_diag_{run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved changed-region diagnostics: {}", metrics_path)
    return result


@app.command()
def main(
    ply_dir: str = typer.Option(
        "outputs/ply",
        "--ply-dir",
        help="Directory containing ply_<num_views>_<iterations>.ply files.",
    ),
    data_path: str = typer.Option(..., "--data-path", "-d", help="Base t0 dataset root."),
    change_type: str = typer.Option(
        "add",
        "--change-type",
        "-c",
        help="Blender change subdirectory (add/delete/move).",
    ),
    timestep: int = typer.Option(1, "--timestep", "-t"),
    out_dir: str = typer.Option(
        "outputs/region_eval",
        "--out-dir",
        help="Directory for per-run rendered/GT/mask images.",
    ),
    metrics_dir: str = typer.Option(
        "outputs/region_diagnostics",
        "--metrics-dir",
        help="Directory for region_diag_<views>_<iterations>.json files.",
    ),
    sh_degree: int = typer.Option(0, "--sh-degree", help="SH degree used during training."),
    white_background: bool = typer.Option(
        False,
        "--white-background/--black-background",
        help="Use the same Blender background convention as training.",
    ),
    mask_threshold: float = typer.Option(
        0.01,
        "--mask-threshold",
        min=0.0,
        max=1.0,
        help="Normalized RGB threshold for fixed t0-vs-t1 change masks.",
    ),
    mask_dilate_px: int = typer.Option(
        0,
        "--mask-dilate-px",
        min=0,
        help="Optional mask dilation radius in pixels. Default 0 keeps the exact GT-difference mask.",
    ),
) -> None:
    """Evaluate all saved PLYs on whole, changed, and unchanged image regions."""
    ply_root = Path(ply_dir)
    ply_paths = sorted(
        [p for p in ply_root.glob("ply_*_*.ply") if _PLY_RE.match(p.name)],
        key=_ply_sort_key,
    )

    if not ply_paths:
        logger.error("No ply_<num_views>_<iterations>.ply files found in {}", ply_root)
        raise SystemExit(1)

    logger.info("Loading fixed held-out t1 test scene from {} / {}", data_path, change_type)
    scene = _load_scene(
        data_path=data_path,
        images="images",
        change_type=change_type,
        timestep=timestep,
        eval_=True,
        white_background=white_background,
    )

    if not scene.is_nerf_synthetic:
        logger.error("Fixed t0-vs-t1 GT region masks are currently implemented for Blender datasets only.")
        raise SystemExit(1)

    if not scene.test_cameras:
        logger.error("No held-out test cameras found in the t1 scene.")
        raise SystemExit(1)

    output_root = Path(out_dir)
    metrics_root = Path(metrics_dir)
    all_results: list[dict[str, object]] = []

    for idx, ply_path in enumerate(ply_paths, start=1):
        logger.info("=" * 72)
        logger.info("Region evaluation {}/{}: {}", idx, len(ply_paths), ply_path.name)
        logger.info("=" * 72)

        result = _evaluate_one_ply(
            ply_path=ply_path,
            scene=scene,
            data_path=data_path,
            timestep=timestep,
            output_root=output_root,
            metrics_dir=metrics_root,
            white_background=white_background,
            sh_degree=sh_degree,
            mask_threshold=mask_threshold,
            mask_dilate_px=mask_dilate_px,
        )
        all_results.append(result)

    metrics_root.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_root / "region_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}")
    print(f"Evaluated {len(all_results)} PLY checkpoints")
    print(f"Per-run diagnostics: {metrics_root}/region_diag_<views>_<iterations>.json")
    print(f"Combined summary:    {summary_path}")
    print(f"Mask previews:       {output_root}/<views>_<iterations>/t{timestep:04d}/*_change_mask.png")
    print(f"{'='*72}")


if __name__ == "__main__":
    app()
