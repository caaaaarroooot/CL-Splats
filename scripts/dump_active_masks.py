import argparse
from pathlib import Path
from typing import cast

import torch
from loguru import logger
from omegaconf import OmegaConf

from clsplats.config import CLSplatsConfig
from clsplats.dataset.dataset_reader import readNerfSyntheticInfo
from clsplats.trainer import CLSplatsTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Dump CL-Splats active Gaussian masks without optimization."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Base Blender scene path, e.g. data/Blender-Levels/Level-1",
    )
    parser.add_argument(
        "--change-type",
        type=str,
        default="add",
        help="Change directory name, e.g. add/delete/move",
    )
    parser.add_argument(
        "--pretrained-ply",
        type=str,
        default="outputs/gaussians_time_30000.ply",
        help="Pretrained t0 Gaussian PLY",
    )
    parser.add_argument(
        "--views",
        type=int,
        nargs="+",
        default=[5, 15, 30, 50],
        help="Numbers of change views to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="View sampling seed",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/active_masks",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional output suffix, e.g. pos1_vis2 -> active_5_pos1_vis2.pt",
    )

    # Coverage / multi-view gate ablation parameters.
    parser.add_argument("--min-visible-views", type=int, default=None)
    parser.add_argument("--min-positive-views", type=int, default=None)
    parser.add_argument("--min-seed-views", type=int, default=None)
    parser.add_argument("--min-positive-ratio", type=float, default=None)
    parser.add_argument("--final-thresh", type=float, default=None)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    yaml_cfg = OmegaConf.load("configs/cl-splats.yaml")
    base_cfg = OmegaConf.structured(CLSplatsConfig)
    cfg = cast(CLSplatsConfig, OmegaConf.merge(base_cfg, yaml_cfg))

    cfg.data_path = args.data_path
    cfg.white_background = True

    # Keep the same train/test split as the previous experiments:
    # 50 change-scene train views + 25 held-out test views.
    cfg.eval = True

    cfg.model.pretrained_ply = args.pretrained_ply
    cfg.train.start_time = 0
    cfg.train.num_times = 2
    cfg.train.view_sample_seed = args.seed

    # Apply lifter gate overrides BEFORE constructing the trainer/lifter.
    if args.min_visible_views is not None:
        cfg.lifter.min_visible_views = args.min_visible_views
    if args.min_positive_views is not None:
        cfg.lifter.min_positive_views = args.min_positive_views
    if args.min_seed_views is not None:
        cfg.lifter.min_seed_views = args.min_seed_views
    if args.min_positive_ratio is not None:
        cfg.lifter.min_positive_ratio = args.min_positive_ratio
    if args.final_thresh is not None:
        cfg.lifter.final_thresh = args.final_thresh

    cfg.history.log_history = False
    # Enable lifter.last_stats so each .pt records exactly which gate removed
    # candidates. train() is never called, so no diagnostic JSON is written.
    cfg.diagnostics.enabled = True
    cfg.wandb_mode = "disabled"

    pretrained = Path(args.pretrained_ply)
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained PLY not found: {pretrained}")

    logger.info(
        "Lifter gates: visible>={} positive>={} seed>={} ratio>={:.3f} final>{:.3f}",
        cfg.lifter.min_visible_views,
        cfg.lifter.min_positive_views,
        cfg.lifter.min_seed_views,
        cfg.lifter.min_positive_ratio,
        cfg.lifter.final_thresh,
    )

    # ---------------------------------------------------------
    # Load t0 scene and construct detector/lifter once
    # ---------------------------------------------------------
    logger.info("Loading base scene: {}", args.data_path)
    base_scene = readNerfSyntheticInfo(
        path=args.data_path,
        white_background=True,
        eval=True,
    )
    trainer = CLSplatsTrainer(cfg, base_scene)

    # ---------------------------------------------------------
    # Load t1 changed scene once
    # ---------------------------------------------------------
    change_path = Path(args.data_path) / args.change_type
    if not change_path.is_dir():
        raise FileNotFoundError(f"Change scene not found: {change_path}")

    logger.info("Loading changed scene: {}", change_path)
    change_scene = readNerfSyntheticInfo(
        path=str(change_path),
        white_background=True,
        eval=True,
    )
    logger.info("Available t1 train views: {}", len(change_scene.train_cameras))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # No optimization: DINO -> depth -> lifting -> active mask only
    # ---------------------------------------------------------
    for num_views in args.views:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Preparing active mask with {} views", num_views)
        logger.info("=" * 70)

        cfg.train.num_change_views = num_views
        trainer.update_cameras(change_scene, timestep=1)
        trainer.prepare_timestep(1)

        if trainer.active_mask is None:
            raise RuntimeError(f"active_mask is None for {num_views} views")

        active_mask = trainer.active_mask.detach().bool().cpu()
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        positions = trainer.gaussians.params.positions.detach().cpu()
        active_positions = positions[active_indices]

        payload = {
            "num_views": int(num_views),
            "seed": int(args.seed),
            "timestep": 1,
            "label": str(args.label),
            "data_path": str(args.data_path),
            "change_type": str(args.change_type),
            "pretrained_ply": str(args.pretrained_ply),
            "selected_views": list(trainer._selected_change_views),
            "lifter_config": {
                "min_visible_views": int(cfg.lifter.min_visible_views),
                "min_positive_views": int(cfg.lifter.min_positive_views),
                "min_seed_views": int(cfg.lifter.min_seed_views),
                "min_positive_ratio": float(cfg.lifter.min_positive_ratio),
                "final_thresh": float(cfg.lifter.final_thresh),
                "k_nn": int(cfg.lifter.k_nn),
                "local_radius_thresh": float(cfg.lifter.local_radius_thresh),
                "depth_tol_abs": float(cfg.lifter.depth_tol_abs),
                "depth_tol_rel": float(cfg.lifter.depth_tol_rel),
            },
            "lifter_stats": dict(trainer.lifter.last_stats),
            "total_gaussians": int(active_mask.numel()),
            "active_count": int(active_mask.sum().item()),
            "active_mask": active_mask,
            "active_indices": active_indices,
            "active_positions": active_positions,
        }

        suffix = f"_{args.label}" if args.label else ""
        output_path = out_dir / f"active_{num_views}{suffix}.pt"
        torch.save(payload, output_path)

        logger.info("Saved: {}", output_path)
        logger.info(
            "Active Gaussians: {} / {} ({:.4f}%)",
            payload["active_count"],
            payload["total_gaussians"],
            100.0 * payload["active_count"] / payload["total_gaussians"],
        )
        logger.info("Selected views: {}", payload["selected_views"])

        mv = trainer.lifter.last_stats.get("multiview_filter", {})
        fs = trainer.lifter.last_stats.get("final_score_filter", {})
        if mv:
            logger.info(
                "Gate counts: visible={} positive={} seed={} ratio={} multiview={} final={}",
                mv.get("visible_passed"),
                mv.get("positive_passed"),
                mv.get("seed_passed"),
                mv.get("ratio_passed"),
                mv.get("multiview_passed"),
                fs.get("passed"),
            )

    logger.info("")
    logger.info("Finished.")
    logger.info("Files saved to: {}", out_dir.resolve())


if __name__ == "__main__":
    main()
