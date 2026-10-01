import argparse
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf
from loguru import logger

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

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    yaml_cfg = OmegaConf.load("configs/cl-splats.yaml")
    base_cfg = OmegaConf.structured(CLSplatsConfig)
    cfg = cast(CLSplatsConfig, OmegaConf.merge(base_cfg, yaml_cfg))

    cfg.data_path = args.data_path
    cfg.white_background = True

    # 중요:
    # 기존 실험처럼 train/test split을 유지해야
    # change scene train views = 50, test views = 25가 된다.
    cfg.eval = True

    cfg.model.pretrained_ply = args.pretrained_ply
    cfg.train.start_time = 0
    cfg.train.num_times = 2
    cfg.train.view_sample_seed = args.seed

    # optimization은 아예 하지 않으므로 history/diagnostics 불필요
    cfg.history.log_history = False
    cfg.diagnostics.enabled = False
    cfg.wandb_mode = "disabled"

    pretrained = Path(args.pretrained_ply)

    if not pretrained.is_file():
        raise FileNotFoundError(
            f"Pretrained PLY not found: {pretrained}"
        )

    # ---------------------------------------------------------
    # Load t0 scene
    # ---------------------------------------------------------
    logger.info("Loading base scene: {}", args.data_path)

    base_scene = readNerfSyntheticInfo(
        path=args.data_path,
        white_background=True,
        eval=True,
    )

    # pretrained PLY를 가진 trainer를 딱 한 번 생성.
    # DINO / Depth Anything도 한 번만 로드한다.
    trainer = CLSplatsTrainer(cfg, base_scene)

    # ---------------------------------------------------------
    # Load t1 changed scene once
    # ---------------------------------------------------------
    change_path = Path(args.data_path) / args.change_type

    if not change_path.is_dir():
        raise FileNotFoundError(
            f"Change scene not found: {change_path}"
        )

    logger.info("Loading changed scene: {}", change_path)

    change_scene = readNerfSyntheticInfo(
        path=str(change_path),
        white_background=True,
        eval=True,
    )

    logger.info(
        "Available t1 train views: {}",
        len(change_scene.train_cameras),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # IMPORTANT:
    # No train() is ever called below.
    # ---------------------------------------------------------
    for num_views in args.views:

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "Preparing active mask with {} views",
            num_views,
        )
        logger.info("=" * 70)

        # update_cameras() 내부에서 seed=42로
        # 기존 실험과 동일하게 random subset 선정
        cfg.train.num_change_views = num_views

        trainer.update_cameras(
            change_scene,
            timestep=1,
        )

        # DINO -> depth -> lifting -> active_mask
        # 여기까지만 실행.
        trainer.prepare_timestep(1)

        if trainer.active_mask is None:
            raise RuntimeError(
                f"active_mask is None for {num_views} views"
            )

        active_mask = (
            trainer.active_mask
            .detach()
            .bool()
            .cpu()
        )

        active_indices = (
            torch.nonzero(active_mask, as_tuple=False)
            .squeeze(1)
        )

        # Optimization 전이므로 baseline Gaussian positions 그대로.
        # 나중에 추가 영역의 공간 분포 분석용.
        positions = (
            trainer.gaussians.params.positions
            .detach()
            .cpu()
        )

        active_positions = positions[active_indices]

        payload = {
            # experiment metadata
            "num_views": int(num_views),
            "seed": int(args.seed),
            "timestep": 1,

            "data_path": str(args.data_path),
            "change_type": str(args.change_type),
            "pretrained_ply": str(args.pretrained_ply),

            # 어떤 사진이 선택됐는지
            "selected_views": list(
                trainer._selected_change_views
            ),

            # Gaussian information
            "total_gaussians": int(
                active_mask.numel()
            ),

            "active_count": int(
                active_mask.sum().item()
            ),

            # 핵심
            "active_mask": active_mask,

            # mask에서 True인 baseline Gaussian index
            "active_indices": active_indices,

            # 해당 Gaussian들의 optimization 전 xyz
            "active_positions": active_positions,
        }

        output_path = (
            out_dir
            / f"active_{num_views}.pt"
        )

        torch.save(
            payload,
            output_path,
        )

        logger.info(
            "Saved: {}",
            output_path,
        )

        logger.info(
            "Active Gaussians: {} / {} ({:.4f}%)",
            payload["active_count"],
            payload["total_gaussians"],
            100.0
            * payload["active_count"]
            / payload["total_gaussians"],
        )

        logger.info(
            "Selected views: {}",
            payload["selected_views"],
        )

    logger.info("")
    logger.info("Finished.")
    logger.info(
        "Files saved to: {}",
        out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
