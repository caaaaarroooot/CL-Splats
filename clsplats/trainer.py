"""CL-Splats trainer.

Orchestrates the full continual-learning Gaussian Splatting pipeline:
detect → lift → constrain → optimise → prune.
"""

import csv
import json
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from random import Random, randint, seed as random_seed
from typing import TYPE_CHECKING, List

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from loguru import logger
from PIL import Image as PILImage

from clsplats.change_detection.dinov2_detector import DinoV2Detector
from clsplats.config import CLSplatsConfig
from clsplats.constraints.primitives import fit_primitives_for_active, union_distance
from clsplats.dataset.cameras import Camera
from clsplats.dataset.dataset_reader import SceneInfo
from clsplats.history import HistoryRecorder, load_raw_ply
from clsplats.lifter.depth_anything_lifter import DepthAnythingLifter
from clsplats.representation.cl_gaussians import CLGaussians, GaussianParams
from clsplats.utils.sh_utils import RGB2SH

if TYPE_CHECKING:
    from clsplats.dataset.dataset_reader import CameraInfo


# ---------------------------------------------------------------------------
# Metric helpers (used when torchmetrics is not installed)
# ---------------------------------------------------------------------------


def _manual_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio for [B, C, H, W] tensors in [0, max_val]."""
    mse = F.mse_loss(pred, gt)
    return 10.0 * torch.log10(torch.tensor(max_val**2) / mse.clamp(min=1e-10))


def _manual_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Structural Similarity Index for [B, C, H, W] tensors.

    A lightweight single-scale implementation without multi-scale weighting.
    """
    B, C, H, W = pred.shape
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * max_val) ** 2, (K2 * max_val) ** 2

    # Gaussian kernel
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g)[None, None].repeat(C, 1, 1, 1)  # [C, 1, k, k]
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(gt, kernel, padding=pad, groups=C)
    sigma_xx = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_x**2
    sigma_yy = F.conv2d(gt * gt, kernel, padding=pad, groups=C) - mu_y**2
    sigma_xy = F.conv2d(pred * gt, kernel, padding=pad, groups=C) - mu_x * mu_y

    ssim_map = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_map /= (mu_x**2 + mu_y**2 + C1) * (sigma_xx + sigma_yy + C2)
    return ssim_map.mean()


def _initial_scales_from_points(
    xyz: torch.Tensor,
    fallback_scale: float,
    min_scale: float = 1e-4,
) -> torch.Tensor:
    """Initialise Gaussian scales from nearest-neighbour point spacing."""
    if xyz.shape[0] < 2:
        return torch.full_like(xyz, max(fallback_scale, min_scale))

    from scipy.spatial import cKDTree

    pts_np = xyz.detach().cpu().numpy()
    distances, _ = cKDTree(pts_np).query(pts_np, k=2, workers=-1)
    nn_dist = torch.as_tensor(distances[:, 1], dtype=xyz.dtype, device=xyz.device)
    nn_dist = nn_dist.clamp_min(min_scale)
    return nn_dist[:, None].repeat(1, 3)


def _photometric_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_dssim: float,
) -> torch.Tensor:
    """3DGS-style photometric loss: L1 mixed with DSSIM."""
    l1 = (rendered - gt).abs().mean()
    if lambda_dssim <= 0:
        return l1

    # The reference computes SSIM on the raw (unclamped) render.
    pred_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
    gt_bchw = gt.permute(2, 0, 1).unsqueeze(0)
    ssim_value = _manual_ssim(pred_bchw, gt_bchw)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_value)


def _load_gt_image(
    image_path: str,
    white_background: bool,
    composite_alpha: bool,
) -> PILImage.Image:
    """Load a ground-truth image, compositing RGBA onto the background colour.

    Mirrors the reference Blender reader: transparent pixels are blended onto
    the training background so supervision covers empty space too.
    """
    img = PILImage.open(image_path)
    if not composite_alpha or img.mode != "RGBA":
        return img
    arr = np.array(img.convert("RGBA")).astype(np.float32) / 255.0
    bg = 1.0 if white_background else 0.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + bg * (1.0 - arr[:, :, 3:4])
    return PILImage.fromarray((rgb * 255.0).astype(np.uint8), "RGB")


_DIAGNOSTIC_FIELDS = [
    "timestamp",
    "phase",
    "data_path",
    "pretrained_ply",
    "timestep",
    "iterations",
    "num_views",
    "available_change_views",
    "requested_change_views",
    "view_sample_seed",
    "selected_views",
    "gaussians_before_prepare",
    "render_sec",
    "dino_sec",
    "change_mask_mean_ratio",
    "change_mask_min_ratio",
    "change_mask_max_ratio",
    "total_views",
    "valid_views",
    "skipped_views",
    "depth_estimation_sec",
    "depth_alignment_sec",
    "lifting_other_sec",
    "lifting_total_sec",
    "active_gaussians",
    "active_gaussian_ratio",
    "prepare_sec",
    "gaussians_before_train",
    "gaussians_after_train",
    "gaussian_count_delta",
    "loss_first",
    "loss_last",
    "loss_mean",
    "loss_min",
    "loss_max",
    "optimization_sec",
    "total_update_sec",
]


class CLSplatsTrainer:
    """Minimal gsplat-backed trainer for continual-learning scene editing.

    Sets up Gaussians from a ``SceneInfo`` point cloud, renders with gsplat,
    and routes images through the DINOv2 change detector.
    """

    def __init__(self, cfg: CLSplatsConfig, scene: SceneInfo):
        self.cfg = cfg

        experiment_seed = int(self.cfg.train.view_sample_seed)

        random_seed(experiment_seed)
        np.random.seed(experiment_seed)
        torch.manual_seed(experiment_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(experiment_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestep = 0
        self.active_mask = None
        self._primitives: list = []
        self._outside_counts = None
        self._global_step = 0
        # Iteration within the current timestep — drives the LR schedule and
        # the gsplat strategy so densification applies at every timestep.
        self._timestep_iter = 0
        self._viewpoint_stack: list[Camera] = []
        self._viewpoint_indices: list[int] = []
        self.scene_extent = float(scene.nerf_normalization.get("radius", 1.0))
        self._is_nerf_synthetic = bool(scene.is_nerf_synthetic)
        self._history = HistoryRecorder()
        self._diagnostics: dict[str, object] = {}
        self._diagnostic_update_start: float | None = None
        self._selected_change_views: list[str] = []
        self._available_change_views: int = 0

        # 1) Initialise Gaussians from scene point cloud
        pcd = scene.point_cloud
        xyz = torch.from_numpy(pcd.points).float().to(self.device)
        rgb = torch.from_numpy(pcd.colors).float().to(self.device)

        n_coeffs = (cfg.model.sh_degree + 1) ** 2
        N = xyz.shape[0]

        sh_features = torch.zeros((N, 3, n_coeffs), device=self.device)
        sh_features[:, :, 0] = RGB2SH(rgb)  # DC term encodes RGB as SH coefficients

        scales = _initial_scales_from_points(xyz, cfg.model.init_scale)
        quats = torch.zeros((N, 4), device=self.device)
        quats[:, 0] = 1.0
        opacity = torch.full((N, 1), cfg.model.init_opacity, device=self.device)

        params = GaussianParams(
            positions=xyz,
            scales=scales,
            quats=quats,
            sh_features=sh_features,
            opacity=opacity,
        )
        # 3DGS scales the position learning rate by the camera extent.
        self.gaussians = CLGaussians(cfg, params, spatial_lr_scale=self.scene_extent)

        baseline_ply = Path(cfg.model.pretrained_ply) if cfg.model.pretrained_ply else None

        if baseline_ply is not None and baseline_ply.is_file():
            logger.info(
                "Using pretrained 30000-iteration 3DGS: {path}",
                path=baseline_ply,
            )

            raw = load_raw_ply(str(baseline_ply))
            sh = torch.cat([raw["sh0"], raw["shN"]], dim=1)

            params = GaussianParams(
                positions=raw["means"],
                scales=torch.exp(raw["scales"]),
                quats=raw["quats"],
                sh_features=sh.permute(0, 2, 1).contiguous(),
                opacity=torch.sigmoid(raw["opacities"]).unsqueeze(-1),
            )

            self.gaussians = CLGaussians(
                cfg,
                params,
                spatial_lr_scale=self.scene_extent,
            )

        self.gaussians.initialize_strategy_state(self.scene_extent)

        # 2) Cameras and change detector
        self.train_cameras: List[Camera] = []
        for uid, cam_info in enumerate(scene.train_cameras):
            self.train_cameras.append(
                self._camera_from_info(cam_info, uid, scene.is_nerf_synthetic)
            )

        # Group cameras by timestep for temporal data
        self._cameras_by_timestep: dict[int, list] = defaultdict(list)
        for cam in self.train_cameras:
            self._cameras_by_timestep[cam.timestep].append(cam)
        self._reset_viewpoint_stack()

        self.detector = DinoV2Detector(cfg.change)
        self.lifter = DepthAnythingLifter(cfg)

    def _sync_cuda(self) -> None:
        if self.cfg.diagnostics.enabled and self.device.type == "cuda":
            torch.cuda.synchronize()

    def _save_diagnostics(self) -> None:
        if not self.cfg.diagnostics.enabled or not self._diagnostics:
            return

        out_dir = Path(self.cfg.diagnostics.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestep = int(self._diagnostics.get("timestep", self.timestep))

        iterations = int(
            self._diagnostics.get(
                "iterations",
                self.cfg.train.iters_per_timestep,
            )
        )

        num_views = int(
            self._diagnostics.get(
                "num_views",
                len(self.train_cameras),
            )
        )

        json_path = out_dir / (f"diag_{num_views}_{iterations}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                self._diagnostics,
                f,
                indent=2,
                ensure_ascii=False,
            )

        csv_path = out_dir / "summary.csv"
        write_header = not csv_path.exists()

        row = {field: self._diagnostics.get(field, "") for field in _DIAGNOSTIC_FIELDS}

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_DIAGNOSTIC_FIELDS,
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        logger.info(
            "Saved diagnostics: {json_path}",
            json_path=json_path,
        )

    def _reset_viewpoint_stack(self) -> None:
        self._viewpoint_stack = self.train_cameras.copy()
        self._viewpoint_indices = list(range(len(self._viewpoint_stack)))

    def _pop_random_train_camera(self) -> Camera:
        if not self._viewpoint_stack:
            self._reset_viewpoint_stack()
        rand_idx = randint(0, len(self._viewpoint_indices) - 1)
        self._viewpoint_indices.pop(rand_idx)
        return self._viewpoint_stack.pop(rand_idx)

    def _camera_from_info(
        self,
        cam_info,
        uid: int,
        is_nerf_synthetic: bool,
        timestep: int | None = None,
    ) -> Camera:
        """Build a Camera from CameraInfo, including optional inverse depth."""
        img = _load_gt_image(
            cam_info.image_path,
            white_background=getattr(self.cfg, "white_background", False),
            composite_alpha=is_nerf_synthetic,
        )
        return Camera(
            resolution=(cam_info.width, cam_info.height),
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=img,
            image_name=cam_info.image_name,
            uid=uid,
            data_device="cpu",
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=cam_info.is_test,
            timestep=cam_info.timestep if timestep is None else timestep,
        )

    def update_cameras(self, scene: SceneInfo, timestep: int) -> None:
        """Load new cameras from *scene* for the given *timestep*.

        Optionally subsamples the change-scene views.  A value of
        ``train.num_change_views == 0`` keeps all available views.
        """
        self._is_nerf_synthetic = bool(scene.is_nerf_synthetic)

        cam_infos = list(scene.train_cameras)
        total_views = len(cam_infos)

        self._available_change_views = total_views

        requested_views = self.cfg.train.num_change_views

        if requested_views < 0:
            raise ValueError(
                "train.num_change_views must be >= 0 (0 means use all available views)."
            )

        if requested_views > total_views:
            raise ValueError(
                f"Requested {requested_views} change views, but only {total_views} are available."
            )

        # 0 or exactly all available views -> no subsampling.
        if 0 < requested_views < total_views:
            rng = Random(self.cfg.train.view_sample_seed)

            # Sample indices, then sort them so the selected subset keeps
            # the original dataset ordering.
            selected_indices = sorted(
                rng.sample(
                    range(total_views),
                    requested_views,
                )
            )

            cam_infos = [cam_infos[i] for i in selected_indices]

            logger.info(
                "Randomly selected {selected}/{total} change views (seed={seed}).",
                selected=len(cam_infos),
                total=total_views,
                seed=self.cfg.train.view_sample_seed,
            )
        else:
            logger.info(
                "Using all {total} available change views.",
                total=total_views,
            )

        new_cameras: list[Camera] = []

        for uid, cam_info in enumerate(cam_infos):
            new_cameras.append(
                self._camera_from_info(
                    cam_info,
                    uid,
                    scene.is_nerf_synthetic,
                    timestep=timestep,
                )
            )

        self._selected_change_views = [cam.image_name for cam in new_cameras]

        self._cameras_by_timestep[timestep] = new_cameras

    def prepare_timestep(self, timestep: int) -> None:
        """Set up the trainer for optimising at the given *timestep*.

        At ``start_time`` all Gaussians are active (standard 3DGS).  From
        ``start_time + 1`` onwards, change detection and lifting select the
        subset to optimise.
        """
        assert timestep < self.cfg.train.num_times, "timestep >= num_times"
        self.timestep = timestep
        self._timestep_iter = 0

        # Select cameras for this timestep (if temporal data is available)
        if timestep in self._cameras_by_timestep:
            self.train_cameras = self._cameras_by_timestep[timestep]
            self._reset_viewpoint_stack()

        if len(self.train_cameras) == 0:
            self.active_mask = None
            return

        start_time = self.cfg.train.start_time
        is_initial = timestep == start_time

        if self.cfg.diagnostics.enabled:
            self._sync_cuda()

            self._diagnostic_update_start = time.perf_counter()

            self._diagnostics = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "phase": ("initial" if is_initial else "continual"),
                "data_path": self.cfg.data_path,
                "pretrained_ply": (self.cfg.model.pretrained_ply),
                "timestep": timestep,
                "num_views": len(self.train_cameras),
                "available_change_views": self._available_change_views,
                "requested_change_views": self.cfg.train.num_change_views,
                "view_sample_seed": self.cfg.train.view_sample_seed,
                "selected_views": self._selected_change_views,
                "gaussians_before_prepare": self.gaussians.num_gaussians,
            }

        # Fresh strategy + state per timestep: the densification window is
        # driven by the per-timestep iteration, and during the CL phase the
        # strategy must never modify inactive Gaussians (no opacity reset,
        # no global opacity pruning) — they are frozen by contract, which
        # also makes history recovery exact.
        self.gaussians.setup_timestep(self.scene_extent, cl_phase=not is_initial)

        if is_initial:
            N = self.gaussians.params.positions.shape[0]
            self.active_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            self._primitives = []
            self._outside_counts = None
            return

        # The representation is already trained at full SH resolution; the
        # progressive activation only applies to the initial reconstruction.
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        # Detect → lift. Rendered depths anchor the lifter's monocular depth
        # to the scene's metric scale.
        change_masks = []
        rendered_depths = []

        render_sec = 0.0
        dino_sec = 0.0
        change_ratios = []

        for cam in self.train_cameras:
            with torch.no_grad():
                # ---------- Existing GS render ----------
                if self.cfg.diagnostics.enabled:
                    self._sync_cuda()
                    render_start = time.perf_counter()

                rendered, rendered_depth = self._render_camera(
                    cam,
                    return_depth=True,
                )

                if self.cfg.diagnostics.enabled:
                    self._sync_cuda()
                    render_sec += time.perf_counter() - render_start

                # ---------- DINO change detection ----------
                if self.cfg.diagnostics.enabled:
                    self._sync_cuda()
                    dino_start = time.perf_counter()

                change_mask_2d = self.detector.predict_change_mask(
                    rendered_image=rendered,
                    observation=(cam.original_image.permute(1, 2, 0).contiguous().to(self.device)),
                )

                if self.cfg.diagnostics.enabled:
                    self._sync_cuda()
                    dino_sec += time.perf_counter() - dino_start

            change_masks.append(change_mask_2d)
            rendered_depths.append(rendered_depth)

            if self.cfg.diagnostics.enabled:
                ratio = float((change_mask_2d > 0.5).float().mean().item())
                change_ratios.append(ratio)

        self.active_mask = self.lifter.lift(
            gaussians=self.gaussians,
            cameras=self.train_cameras,
            change_masks=change_masks,
            rendered_depths=rendered_depths,
        )

        if self.cfg.diagnostics.enabled:
            total_gaussians = self.gaussians.num_gaussians

            active_gaussians = int(self.active_mask.sum().item())

            self._diagnostics.update(
                {
                    "render_sec": render_sec,
                    "dino_sec": dino_sec,
                    "change_mask_mean_ratio": (
                        sum(change_ratios) / len(change_ratios) if change_ratios else 0.0
                    ),
                    "change_mask_min_ratio": (min(change_ratios) if change_ratios else 0.0),
                    "change_mask_max_ratio": (max(change_ratios) if change_ratios else 0.0),
                    "active_gaussians": active_gaussians,
                    "active_gaussian_ratio": (
                        active_gaussians / total_gaussians if total_gaussians > 0 else 0.0
                    ),
                    "experiment_seed": int(self.cfg.train.view_sample_seed),
                    **self.lifter.last_stats,
                }
            )

        # Inactive Gaussians must stay exactly frozen: zero their Adam
        # moments so leftover momentum from earlier timesteps cannot move
        # them despite masked gradients.
        self.gaussians.reset_inactive_optimizer_state(self.active_mask)

        # Snapshot the active rows before any optimisation — together with
        # the end-of-timestep active mask this is the complete delta needed
        # to recover the previous scene state exactly.
        if self.cfg.history.log_history and self.active_mask is not None:
            self._history.begin_timestep(timestep, self.active_mask, self.gaussians.strategy_params)

        # Fit geometric primitives around active Gaussians
        if self.active_mask is not None and self.active_mask.any():
            self._primitives = [
                prim
                for _, prim in fit_primitives_for_active(
                    positions=self.gaussians.params.positions.detach(),
                    active_mask=self.active_mask.detach(),
                    radius_frac=self.cfg.constraints.group_radius_frac,
                )
            ]
        else:
            self._primitives = []

        if self.cfg.diagnostics.enabled and self._diagnostic_update_start is not None:
            self._sync_cuda()

            self._diagnostics["prepare_sec"] = time.perf_counter() - self._diagnostic_update_start

    def _render_camera(self, cam: Camera, return_info: bool = False, return_depth: bool = False):
        """Render a single camera view using gsplat rasterisation.

        With ``return_depth=True`` the expected depth map [H, W] is returned
        alongside the image (0 where nothing was rendered).
        """
        device = self.device
        means = self.gaussians.params.positions.to(device)
        quats = self.gaussians.params.quats.to(device)
        scales = self.gaussians.params.scales.to(device)
        opacities = self.gaussians.params.opacity.squeeze(-1).to(device)

        # SH features: [N, 3, K] → gsplat expects [N, K, 3]
        sh_feats = self.gaussians.params.sh_features.to(device)
        colors = sh_feats.permute(0, 2, 1).contiguous()

        # Camera stores the 3DGS rasterizer convention (transposed / column
        # major); gsplat expects a plain row-major world-to-camera matrix.
        viewmats = cam.world_view_transform.transpose(0, 1).to(device).unsqueeze(0)
        Ks = torch.zeros(1, 3, 3, device=device, dtype=means.dtype)
        Ks[..., 0, 0] = cam.fx
        Ks[..., 1, 1] = cam.fy
        Ks[..., 0, 2] = cam.cx
        Ks[..., 1, 2] = cam.cy
        Ks[..., 2, 2] = 1.0

        width = cam.image_width
        height = cam.image_height

        # Match the GT compositing background (reference renders Blender
        # scenes onto the configured background colour; gsplat defaults to
        # black when no background is given).
        backgrounds = None
        if getattr(self.cfg, "white_background", False):
            backgrounds = torch.ones(1, 3, device=device, dtype=means.dtype)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.gaussians.active_sh_degree,
            backgrounds=backgrounds,
            packed=False,
            distributed=False,
            render_mode="RGB+ED" if return_depth else "RGB",
        )

        img = render_colors[0, ..., :3]  # [H, W, 3]
        if return_depth:
            depth = render_colors[0, ..., 3]  # [H, W] expected depth
            if return_info:
                return img.contiguous(), depth.contiguous(), info
            return img.contiguous(), depth.contiguous()
        if return_info:
            return img.contiguous(), info
        return img.contiguous()

    def _train_step(self, cam: Camera) -> dict:
        """Execute one optimisation step for a single camera view.

        Computes photometric loss, optionally adds a geometric constraint
        loss, masks gradients for inactive Gaussians, steps the optimiser,
        and optionally prunes outliers.

        Returns:
            Dictionary with ``"loss"`` key.
        """
        start_time = self.cfg.train.start_time
        iteration = self._timestep_iter + 1
        self.gaussians.update_learning_rate(iteration)
        if self.timestep == start_time:
            # Progressive SH only applies to the initial reconstruction; at
            # later timesteps the model already carries trained SH features.
            self.gaussians.update_sh_degree(iteration)

        old_count = self.gaussians.num_gaussians
        rendered, raster_info = self._render_camera(cam, return_info=True)
        self.gaussians.step_pre_backward(self._timestep_iter, raster_info)
        gt = cam.original_image.permute(1, 2, 0).contiguous().to(self.device)

        if cam.alpha_mask is not None:
            alpha_mask = cam.alpha_mask.permute(1, 2, 0).contiguous().to(self.device)
            rendered = rendered * alpha_mask

        photometric_loss = _photometric_loss(
            rendered,
            gt,
            lambda_dssim=self.cfg.train.lambda_dssim,
        )

        total_loss = photometric_loss
        if (
            self.timestep > start_time
            and self.active_mask is not None
            and self._primitives
            and self.cfg.constraints.lambda_bound > 0
        ):
            # Full N×P distance — only worth computing when the bound loss
            # actually contributes.
            d_union_full = union_distance(self.gaussians.params.positions, self._primitives)
            mask = self.active_mask.to(self.device)
            if mask.any():
                loss_bound = d_union_full[mask].mean()
                total_loss = photometric_loss + self.cfg.constraints.lambda_bound * loss_bound
        total_loss.backward()

        # Zero gradients for inactive Gaussians (from t1+)
        if self.timestep > start_time and self.active_mask is not None:
            self.gaussians.mask_inactive_gradients(self.active_mask)

        # The strategy may duplicate/split/remove Gaussians; the CL masks are
        # carried through gsplat's ops and returned realigned.
        self.active_mask, self._outside_counts = self.gaussians.step_post_backward(
            self._timestep_iter, raster_info, self.active_mask, self._outside_counts
        )
        self.gaussians.step_optimizer()
        self._sync_after_strategy(old_count)

        # Hard pruning with hysteresis
        if self.timestep > start_time and self.active_mask is not None and self._primitives:
            self._constraint_prune_step(iteration)

        with torch.no_grad():
            pred_bchw = rendered.detach().permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gt_bchw = gt.detach().permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

            psnr = float(_manual_psnr(pred_bchw, gt_bchw))
            ssim = float(_manual_ssim(pred_bchw, gt_bchw))

        self._timestep_iter += 1
        self._global_step += 1

        return {
            "loss": float(total_loss.detach().cpu()),
            "psnr": psnr,
            "ssim": ssim,
        }

    def _constraint_prune_step(self, iteration: int) -> None:
        """Prune *active* Gaussians that stay outside the fitted primitives.

        Only the locally optimised (active) set may ever be pruned — the
        frozen remainder of the scene must be preserved unconditionally,
        regardless of its distance to the change region.
        """
        prune_every = self.cfg.constraints.prune_every
        prune_dist = self.cfg.constraints.prune_dist_thresh
        prune_consec = self.cfg.constraints.prune_consecutive

        if iteration % prune_every != 0:
            return

        N = self.gaussians.params.positions.shape[0]
        if self._outside_counts is None or self._outside_counts.shape[0] != N:
            self._outside_counts = torch.zeros(N, dtype=torch.int64, device=self.device)
        # Recompute on the current Gaussians — the strategy may have
        # changed the count since the loss-time distance was taken.
        with torch.no_grad():
            d_union_now = union_distance(self.gaussians.params.positions.detach(), self._primitives)
        active = self.active_mask.to(self.device)
        outside = (d_union_now > prune_dist) & active
        self._outside_counts[outside] += 1
        self._outside_counts[~outside] = 0
        prune_mask = self._outside_counts >= prune_consec
        # gsplat's global opacity pruning is disabled in the CL phase (it
        # would touch frozen Gaussians); apply it here to the active set.
        with torch.no_grad():
            low_opacity = self.gaussians.params.opacity.squeeze(-1) < (
                self.cfg.train.densify_prune_opa
            )
        prune_mask = prune_mask | (low_opacity & active)
        if prune_mask.any():
            keep = self.gaussians.prune_gaussians(prune_mask)
            self.active_mask = self.active_mask[keep]
            self._outside_counts = self._outside_counts[keep]
            # Re-fit primitives on remaining active Gaussians
            self._refit_primitives()

    def _sync_after_strategy(self, old_count: int) -> None:
        """React to gsplat DefaultStrategy changing the Gaussian count.

        The CL masks themselves are kept aligned by ``step_post_backward``
        (they travel through gsplat's ops); only the fitted primitives need
        refreshing here.
        """
        new_count = self.gaussians.num_gaussians
        if new_count == old_count:
            return

        if self.active_mask is None:
            self.active_mask = torch.ones(new_count, dtype=torch.bool, device=self.device)

        if self.timestep > self.cfg.train.start_time:
            self._refit_primitives()

    def _refit_primitives(self) -> None:
        if self.active_mask is not None and self.active_mask.any():
            self._primitives = [
                prim
                for _, prim in fit_primitives_for_active(
                    positions=self.gaussians.params.positions.detach(),
                    active_mask=self.active_mask.detach(),
                    radius_frac=self.cfg.constraints.group_radius_frac,
                )
            ]
        else:
            self._primitives = []

    def train(self, test_cameras=None) -> None:
        """Run the training loop for the current timestep."""

        if self.timestep == self.cfg.train.start_time:
            num_iters = 30000
        else:
            num_iters = self.cfg.train.iters_per_timestep

        log_interval = self.cfg.train.log_interval

        losses: list[float] = []
        diagnostic_iterations = {1, 10, 50, 100, 500, 1000, 3000, 5000}
        optimization_history: list[dict] = []

        gaussians_before_train = self.gaussians.num_gaussians

        if self.cfg.diagnostics.enabled:
            self._sync_cuda()
            optimization_start = time.perf_counter()

        for it in range(num_iters):
            cam = self._pop_random_train_camera()
            stats = self._train_step(cam)

            if self.cfg.diagnostics.enabled:
                losses.append(stats["loss"])

                current_iteration = it + 1

                if current_iteration in diagnostic_iterations or current_iteration == num_iters:
                    optimization_history.append(
                        {
                            "iteration": current_iteration,
                            "loss": stats["loss"],
                            "psnr": stats["psnr"],
                            "ssim": stats["ssim"],
                            "gaussian_count": self.gaussians.num_gaussians,
                        }
                    )

            if (it + 1) % log_interval == 0:
                logger.info(
                    "[time={time} it={it}/{num_iters}] loss={loss:.4f}",
                    time=self.timestep,
                    it=it + 1,
                    num_iters=num_iters,
                    loss=stats["loss"],
                )

        if self.cfg.diagnostics.enabled:
            self._sync_cuda()
            optimization_sec = time.perf_counter() - optimization_start

        # Final evaluation on all training views
        final_train_psnr_values: list[float] = []
        final_train_ssim_values: list[float] = []

        if self.cfg.diagnostics.enabled:
            self._sync_cuda()
            final_train_eval_start = time.perf_counter()

            with torch.no_grad():
                for eval_cam in self.train_cameras:
                    eval_rendered = self._render_camera(eval_cam)

                    eval_gt = eval_cam.original_image.permute(1, 2, 0).contiguous().to(self.device)

                    if eval_cam.alpha_mask is not None:
                        eval_alpha = (
                            eval_cam.alpha_mask.permute(1, 2, 0).contiguous().to(self.device)
                        )
                        eval_rendered = eval_rendered * eval_alpha

                    pred_bchw = eval_rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                    gt_bchw = eval_gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

                    final_train_psnr_values.append(float(_manual_psnr(pred_bchw, gt_bchw)))
                    final_train_ssim_values.append(float(_manual_ssim(pred_bchw, gt_bchw)))

        final_train_psnr = (
            sum(final_train_psnr_values) / len(final_train_psnr_values)
            if final_train_psnr_values
            else None
        )

        final_train_ssim = (
            sum(final_train_ssim_values) / len(final_train_ssim_values)
            if final_train_ssim_values
            else None
        )

        if self.cfg.diagnostics.enabled:
            self._sync_cuda()
            final_train_eval_sec = time.perf_counter() - final_train_eval_start

        # 여기부터 12번 코드
        if self.cfg.diagnostics.enabled:
            gaussians_after_train = self.gaussians.num_gaussians

            self._diagnostics.update(
                {
                    "iterations": num_iters,
                    "optimization_history": optimization_history,
                    "final_train_psnr": final_train_psnr,
                    "final_train_ssim": final_train_ssim,
                    "final_train_eval_views": len(final_train_psnr_values),
                    "final_train_eval_sec": final_train_eval_sec,
                    "gaussians_before_train": gaussians_before_train,
                    "gaussians_after_train": gaussians_after_train,
                    "gaussian_count_delta": (gaussians_after_train - gaussians_before_train),
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "loss_mean": (sum(losses) / len(losses) if losses else None),
                    "loss_min": min(losses) if losses else None,
                    "loss_max": max(losses) if losses else None,
                    "optimization_sec": optimization_sec,
                }
            )

            if self._diagnostic_update_start is not None:
                self._diagnostics["total_update_sec"] = (
                    time.perf_counter() - self._diagnostic_update_start
                )

        # 기존 history 코드
        if (
            self.cfg.history.log_history
            and self._history.records
            and self._history.records[-1].timestep == self.timestep
            and self._history.records[-1].active_end_mask is None
            and self.active_mask is not None
        ):
            self._history.end_timestep(self.active_mask)

            # Final evaluation on fixed held-out test views
        if self.cfg.diagnostics.enabled and test_cameras:
            self._sync_cuda()
            final_test_eval_start = time.perf_counter()

            test_metrics = self.evaluate(
                test_cameras=test_cameras,
                timestep=self.timestep,
            )

            self._sync_cuda()
            final_test_eval_sec = time.perf_counter() - final_test_eval_start

            self._diagnostics.update(
                {
                    "final_test_psnr": test_metrics["psnr"],
                    "final_test_ssim": test_metrics["ssim"],
                    "final_test_eval_views": len(test_cameras),
                    "final_test_eval_sec": final_test_eval_sec,
                }
            )

        # 마지막에 저장
        if self.cfg.diagnostics.enabled:
            self._save_diagnostics()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cameras: List["CameraInfo"],  # CameraInfo from dataset_reader
        timestep: int,
        out_dir: str | Path = "outputs/eval",
    ) -> dict:
        """Render *test_cameras* and compute image-quality metrics.

        Computes per-image **PSNR** and **SSIM** (via torchmetrics when
        available, with a manual fallback) and saves rendered / ground-truth
        images to *out_dir*.  Summary metrics are logged to W&B if a run is
        active.

        Args:
            test_cameras: Held-out cameras (``CameraInfo`` namedtuples from the
                dataset reader).  They carry ``image_path`` for ground truth.
            timestep: Which training timestep is being evaluated (for logging).
            out_dir: Directory to save rendered images.

        Returns:
            Dict with keys ``"psnr"`` and ``"ssim"`` (mean values over all
            test cameras).
        """
        out_dir = Path(out_dir) / f"t{timestep:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Try to import torchmetrics; fall back to manual implementations.
        try:
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )

            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            use_torchmetrics = True
        except ImportError:
            logger.warning("torchmetrics not installed — using manual PSNR/SSIM fallbacks.")
            use_torchmetrics = False

        psnr_values: list[float] = []
        ssim_values: list[float] = []

        for cam_info in test_cameras:
            # Build a Camera object from the CameraInfo namedtuple. Blender
            # GT must be composited onto the training background (a plain
            # RGB convert would keep junk colours under alpha=0).
            img_pil = _load_gt_image(
                cam_info.image_path,
                white_background=getattr(self.cfg, "white_background", False),
                composite_alpha=getattr(self, "_is_nerf_synthetic", False),
            ).convert("RGB")
            cam = Camera(
                resolution=(cam_info.width, cam_info.height),
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                image=img_pil,
                image_name=cam_info.image_name,
                uid=cam_info.uid,
                data_device="cpu",
                train_test_exp=False,
                is_test_dataset=True,
                is_test_view=True,
                timestep=cam_info.timestep,
            )

            with torch.no_grad():
                rendered = self._render_camera(cam)  # [H, W, 3] in [0, 1]

            gt = cam.original_image.permute(1, 2, 0).to(self.device)  # [H, W, 3]

            # torchmetrics expects [B, C, H, W]
            pred_bchw = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gt_bchw = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

            if use_torchmetrics:
                psnr = float(psnr_metric(pred_bchw, gt_bchw))
                ssim = float(ssim_metric(pred_bchw, gt_bchw))
            else:
                psnr = float(_manual_psnr(pred_bchw, gt_bchw))
                ssim = float(_manual_ssim(pred_bchw, gt_bchw))

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            logger.info(
                "[eval t={t}] {name}: PSNR={psnr:.2f} dB  SSIM={ssim:.4f}",
                t=timestep,
                name=cam_info.image_name,
                psnr=psnr,
                ssim=ssim,
            )

            # Save rendered and ground-truth images
            rendered_np = (rendered.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            gt_np = (gt.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            PILImage.fromarray(rendered_np).save(out_dir / f"{cam_info.image_name}_render.png")
            PILImage.fromarray(gt_np).save(out_dir / f"{cam_info.image_name}_gt.png")

        mean_psnr = float(sum(psnr_values) / len(psnr_values)) if psnr_values else 0.0
        mean_ssim = float(sum(ssim_values) / len(ssim_values)) if ssim_values else 0.0

        logger.info(
            "[eval t={t}] Mean PSNR={psnr:.2f} dB  Mean SSIM={ssim:.4f}  ({n} views)  → {dir}",
            t=timestep,
            psnr=mean_psnr,
            ssim=mean_ssim,
            n=len(psnr_values),
            dir=out_dir,
        )

        # W&B logging
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        f"eval/t{timestep}/psnr": mean_psnr,
                        f"eval/t{timestep}/ssim": mean_ssim,
                    }
                )
                # Log a grid of up to 8 renders
                panels = []
                for cam_info in test_cameras[:8]:
                    rpath = out_dir / f"{cam_info.image_name}_render.png"
                    if rpath.exists():
                        panels.append(wandb.Image(str(rpath), caption=cam_info.image_name))
                if panels:
                    wandb.log({f"eval/t{timestep}/renders": panels})
        except Exception:  # pylint: disable=broad-except
            pass

        return {"psnr": mean_psnr, "ssim": mean_ssim}

    def log_history(self) -> None:
        """Export Gaussians and log metrics."""
        # We export the point cloud to the current working directory, which
        # will typically be managed by Hydra's output directory system, or ./outputs
        out_dir = Path("outputs/ply")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Save a .ply file using the same naming rule as diagnostics
        num_views = len(self.train_cameras)

        if self.timestep == self.cfg.train.start_time:
            iterations = 30000
        else:
            iterations = self.cfg.train.iters_per_timestep

        ply_path = out_dir / f"ply_{num_views}_{iterations}.ply"

        self.gaussians.export_ply(str(ply_path))
        logger.info("Exported optimized Gaussians to {path}", path=ply_path)

        # Persist the per-timestep deltas for cl-splats-history.
        if self._history.records:
            history_dir = out_dir / "history"
            self._history.save(history_dir)
            logger.info("Saved history records to {path}", path=history_dir)
