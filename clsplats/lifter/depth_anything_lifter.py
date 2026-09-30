"""Depth-Anything V2 lifter.

Estimates monocular depth using Depth-Anything V2 from HuggingFace and
lifts 2-D change masks into a per-Gaussian change mask using multi-view
depth back-projection and Gaussian proximity scoring.

Diagnostic logging additionally records:
- per-view depth-alignment quality/failure reason
- positive-pixel sampling statistics
- kNN/local-radius/depth-consistency gate statistics
- multi-view consistency statistics
- final lifting-score statistics
"""

from typing import Optional
import time

import numpy as np
import torch
from loguru import logger
from PIL import Image as PILImage

from clsplats.config import CLSplatsConfig
from clsplats.dataset.cameras import Camera
from clsplats.lifter.base_lifter import BaseLifter
from clsplats.representation.cl_gaussians import CLGaussians
from clsplats.utils.custom_types import Image


def align_relative_depth(
    mono: torch.Tensor,
    rendered_depth: torch.Tensor,
    change_mask: torch.Tensor,
    min_pixels: int = 200,
) -> tuple[Optional[torch.Tensor], dict]:
    """Anchor relative monocular depth to the scene's metric scale.

    Depth-Anything predicts affine-invariant disparity. Fit

        1 / z ~= a * mono + b

    using unchanged pixels covered by the existing rendering.

    Returns:
        aligned_depth:
            Metric-aligned depth, or None when alignment fails.

        diagnostics:
            Dictionary containing:
            - success
            - anchor_pixels
            - r2
            - scale_a
            - bias_b
            - failure_reason
    """

    valid = (
        (~change_mask)
        & (rendered_depth > 1e-6)
        & torch.isfinite(rendered_depth)
        & torch.isfinite(mono)
    )

    anchor_pixels = int(valid.sum().item())

    if anchor_pixels < min_pixels:
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": None,
            "scale_a": None,
            "bias_b": None,
            "failure_reason": "insufficient_anchor_pixels",
        }

    x = mono[valid].float()
    y = 1.0 / rendered_depth[valid].float()

    ones = torch.ones_like(x)
    A = torch.stack([x, ones], dim=-1)

    try:
        solution = torch.linalg.lstsq(
            A,
            y.unsqueeze(-1),
        ).solution.squeeze(-1)
    except Exception as exc:
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": None,
            "scale_a": None,
            "bias_b": None,
            "failure_reason": f"lstsq_failed:{type(exc).__name__}",
        }

    a = float(solution[0].item())
    b = float(solution[1].item())

    if not np.isfinite(a) or not np.isfinite(b):
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": None,
            "scale_a": a if np.isfinite(a) else None,
            "bias_b": b if np.isfinite(b) else None,
            "failure_reason": "non_finite_alignment_parameters",
        }

    prediction = a * x + b

    ss_res = ((prediction - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)

    r2 = float((1.0 - ss_res / ss_tot).item())

    if not np.isfinite(r2):
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": None,
            "scale_a": a,
            "bias_b": b,
            "failure_reason": "non_finite_r2",
        }

    if a <= 0:
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": r2,
            "scale_a": a,
            "bias_b": b,
            "failure_reason": "non_positive_scale",
        }

    if r2 < 0.3:
        return None, {
            "success": False,
            "anchor_pixels": anchor_pixels,
            "r2": r2,
            "scale_a": a,
            "bias_b": b,
            "failure_reason": "low_r2",
        }

    disparity = (a * mono + b).clamp_min(1e-6)
    aligned_depth = 1.0 / disparity

    return aligned_depth, {
        "success": True,
        "anchor_pixels": anchor_pixels,
        "r2": r2,
        "scale_a": a,
        "bias_b": b,
        "failure_reason": None,
    }


class DepthAnythingLifter(BaseLifter):
    """Lifter that uses Depth-Anything V2 for monocular depth estimation."""

    def __init__(self, cfg: CLSplatsConfig):
        super().__init__(cfg)

        from transformers import pipeline as hf_pipeline

        model_name = cfg.lifter.depth_model
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=model_name,
        )

        lcfg = cfg.lifter

        self.k_nn = lcfg.k_nn
        self.local_radius_thresh = lcfg.local_radius_thresh
        self.depth_tol_abs = lcfg.depth_tol_abs
        self.depth_tol_rel = lcfg.depth_tol_rel

        self.lambda_seed = lcfg.lambda_seed
        self.lambda_neg = lcfg.lambda_neg

        self.min_visible_views = lcfg.min_visible_views
        self.min_positive_views = lcfg.min_positive_views
        self.min_seed_views = lcfg.min_seed_views
        self.min_positive_ratio = lcfg.min_positive_ratio

        self.final_thresh = lcfg.final_thresh

        self.diagnostics_enabled = cfg.diagnostics.enabled

        self.last_stats: dict = {}

    def _sync_cuda(self) -> None:
        if self.diagnostics_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def estimate_depth(
        self,
        observation: Image,
    ) -> torch.Tensor:
        """Estimate relative depth from observation image [H, W, 3].

        Returns:
            float32 tensor [H, W] normalized approximately to [0, 1].
        """

        obs_np = (observation.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")

        pil_img = PILImage.fromarray(obs_np)

        depth_output = self._pipe(pil_img)["depth"]

        depth_np = np.array(
            depth_output,
            dtype=np.float32,
        )

        depth_tensor = torch.from_numpy(depth_np)
        depth_tensor = depth_tensor / 255.0

        return depth_tensor

    @torch.no_grad()
    def lift(
        self,
        gaussians: CLGaussians,
        cameras: "list[Camera]",
        change_masks: "list[torch.Tensor]",
        rendered_depths: "Optional[list[torch.Tensor]]" = None,
    ) -> torch.Tensor:
        """Lift 2-D change evidence into Gaussian-space change evidence.

        Pipeline:
            change pixels
            -> monocular depth
            -> metric depth alignment
            -> 3-D back-projection
            -> kNN Gaussian candidates
            -> local-radius filtering
            -> depth-consistency filtering
            -> positive/negative evidence
            -> multi-view consistency
            -> final score threshold
        """

        device = gaussians.params.positions.device

        N = gaussians.params.positions.shape[0]

        self.last_stats = {}

        # ---------------------------------------------------------
        # Timing diagnostics
        # ---------------------------------------------------------

        depth_estimation_sec = 0.0
        depth_alignment_sec = 0.0

        if self.diagnostics_enabled:
            self._sync_cuda()
            lift_start = time.perf_counter()
        else:
            lift_start = 0.0

        # ---------------------------------------------------------
        # Gaussian evidence
        # ---------------------------------------------------------

        seed_score = torch.zeros(
            N,
            device=device,
        )

        seed_votes = torch.zeros(
            N,
            device=device,
        )

        neg_score = torch.zeros(
            N,
            device=device,
        )

        neg_votes = torch.zeros(
            N,
            device=device,
        )

        visible_views = torch.zeros(
            N,
            dtype=torch.int32,
            device=device,
        )

        positive_views = torch.zeros(
            N,
            dtype=torch.int32,
            device=device,
        )

        seed_views = torch.zeros(
            N,
            dtype=torch.int32,
            device=device,
        )

        means = gaussians.params.positions
        scales = gaussians.params.scales

        # ---------------------------------------------------------
        # New diagnostics
        # ---------------------------------------------------------

        skipped_views = 0

        depth_alignment_per_view = []
        lifting_per_view = []

        total_positive_pixels = 0
        sampled_positive_pixels = 0

        total_knn_candidates = 0

        local_radius_pass = 0
        depth_consistency_pass = 0

        gaussians_with_positive_evidence_set = set()

        # ---------------------------------------------------------
        # Process each view
        # ---------------------------------------------------------

        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            obs = cam.original_image.permute(1, 2, 0).contiguous()

            view_name = getattr(
                cam,
                "image_name",
                str(view_id),
            )

            # Per-view counters
            view_positive_pixels = 0
            view_sampled_positive_pixels = 0
            view_knn_candidates = 0
            view_local_radius_pass = 0
            view_depth_consistency_pass = 0

            # -----------------------------------------------------
            # Depth estimation
            # -----------------------------------------------------

            if self.diagnostics_enabled:
                self._sync_cuda()
                depth_start = time.perf_counter()

            depth = self.estimate_depth(obs).to(device)

            if self.diagnostics_enabled:
                self._sync_cuda()
                depth_estimation_sec += time.perf_counter() - depth_start

            mask = mask.to(device)

            # -----------------------------------------------------
            # Metric depth alignment
            # -----------------------------------------------------

            if rendered_depths is not None:
                if self.diagnostics_enabled:
                    self._sync_cuda()
                    alignment_start = time.perf_counter()

                aligned, alignment_info = align_relative_depth(
                    mono=depth,
                    rendered_depth=rendered_depths[view_id].to(device),
                    change_mask=mask > 0.5,
                )

                if self.diagnostics_enabled:
                    self._sync_cuda()
                    depth_alignment_sec += time.perf_counter() - alignment_start

                alignment_info["view_id"] = int(view_id)
                alignment_info["view_name"] = str(view_name)

                depth_alignment_per_view.append(alignment_info)

                if aligned is None:
                    skipped_views += 1

                    lifting_per_view.append(
                        {
                            "view_id": int(view_id),
                            "view_name": str(view_name),
                            "skipped": True,
                            "skip_reason": alignment_info["failure_reason"],
                            "positive_pixels": 0,
                            "sampled_positive_pixels": 0,
                            "knn_candidates": 0,
                            "local_radius_pass": 0,
                            "local_radius_pass_ratio": 0.0,
                            "depth_consistency_pass": 0,
                            "depth_consistency_pass_ratio": 0.0,
                        }
                    )

                    continue

                depth = aligned

            else:
                depth_alignment_per_view.append(
                    {
                        "view_id": int(view_id),
                        "view_name": str(view_name),
                        "success": True,
                        "anchor_pixels": None,
                        "r2": None,
                        "scale_a": None,
                        "bias_b": None,
                        "failure_reason": None,
                        "alignment_used": False,
                    }
                )

            # -----------------------------------------------------
            # Positive pixels
            # -----------------------------------------------------

            pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)

            if pos_pixels.any():
                ys, xs = torch.nonzero(
                    pos_pixels,
                    as_tuple=True,
                )

                original_positive_count = int(ys.numel())

                view_positive_pixels = original_positive_count

                total_positive_pixels += original_positive_count

                # Limit number of positive pixels to avoid
                # excessive M x N cdist memory.
                max_pos = min(
                    ys.numel(),
                    2048,
                )

                if ys.numel() > max_pos:
                    perm = torch.randperm(
                        ys.numel(),
                        device=device,
                    )[:max_pos]

                    ys = ys[perm]
                    xs = xs[perm]

                sampled_count = int(ys.numel())

                view_sampled_positive_pixels = sampled_count

                sampled_positive_pixels += sampled_count

                d = depth[ys, xs]

                # -------------------------------------------------
                # Back-project changed pixels into world space
                # -------------------------------------------------

                x_cam = (xs.float() - cam.cx) / cam.fx * d

                y_cam = (ys.float() - cam.cy) / cam.fy * d

                z_cam = d

                ones = torch.ones_like(z_cam)

                p_cam = torch.stack(
                    [
                        x_cam,
                        y_cam,
                        z_cam,
                        ones,
                    ],
                    dim=-1,
                )

                Twc = cam.Twc.to(device)

                p_world_h = p_cam @ Twc

                p_world = p_world_h[..., :3] / p_world_h[..., 3:]

                # -------------------------------------------------
                # kNN Gaussian candidates
                # -------------------------------------------------

                dists = torch.cdist(
                    p_world,
                    means,
                )

                knn_dists, knn_idx = torch.topk(
                    dists,
                    k=min(
                        self.k_nn,
                        N,
                    ),
                    dim=-1,
                    largest=False,
                )

                del dists

                candidate_count = int(knn_idx.numel())

                view_knn_candidates = candidate_count

                total_knn_candidates += candidate_count

                # -------------------------------------------------
                # Local scale-aware radius gate
                # -------------------------------------------------

                local_scales = scales[knn_idx]

                denom = local_scales.norm(dim=-1) + 1e-6

                d_local = knn_dists / denom

                valid = d_local < self.local_radius_thresh

                local_pass_count = int(valid.sum().item())

                view_local_radius_pass = local_pass_count

                local_radius_pass += local_pass_count

                if valid.any():
                    # ---------------------------------------------
                    # Depth-consistency gate
                    # ---------------------------------------------

                    Tcw = torch.inverse(Twc)

                    knn_means = means[knn_idx]

                    M, k = knn_means.shape[:2]

                    knn_means_h = torch.cat(
                        [
                            knn_means,
                            torch.ones(
                                M,
                                k,
                                1,
                                device=device,
                            ),
                        ],
                        dim=-1,
                    )

                    knn_cam = knn_means_h @ Tcw

                    z_knn = knn_cam[
                        ...,
                        2,
                    ]

                    depth_pix = d.unsqueeze(-1)

                    depth_ok = (z_knn - depth_pix).abs() < (
                        self.depth_tol_abs + self.depth_tol_rel * depth_pix
                    )

                    valid_final = valid & depth_ok

                    depth_pass_count = int(valid_final.sum().item())

                    view_depth_consistency_pass = depth_pass_count

                    depth_consistency_pass += depth_pass_count

                    if valid_final.any():
                        d_local_valid = d_local.masked_fill(
                            ~valid_final,
                            1e9,
                        )

                        weights = torch.exp(-0.5 * d_local_valid**2)

                        weights_sum = (
                            weights.sum(
                                dim=-1,
                                keepdim=True,
                            )
                            + 1e-8
                        )

                        weights = weights / weights_sum

                        mask_vals = mask[ys, xs].unsqueeze(-1).float()

                        contrib = mask_vals * weights

                        flat_idx = knn_idx.reshape(-1)

                        flat_contrib = contrib.reshape(-1)

                        flat_valid = valid_final.reshape(-1)

                        flat_idx = flat_idx[flat_valid]

                        flat_contrib = flat_contrib[flat_valid]

                        seed_score.index_add_(
                            0,
                            flat_idx,
                            flat_contrib,
                        )

                        seed_votes.index_add_(
                            0,
                            flat_idx,
                            flat_contrib,
                        )

                        affected = torch.unique(flat_idx)

                        positive_views[affected] += 1

                        seed_views[affected] += 1

                        visible_views[affected] += 1

                        # Only diagnostic bookkeeping.
                        if self.diagnostics_enabled:
                            gaussians_with_positive_evidence_set.update(
                                affected.detach().cpu().tolist()
                            )

            # -----------------------------------------------------
            # Weak negatives
            # -----------------------------------------------------

            neg_pixels = (~pos_pixels) & torch.isfinite(depth) & (depth > 0)

            if neg_pixels.any():
                ys_n, xs_n = torch.nonzero(
                    neg_pixels,
                    as_tuple=True,
                )

                max_neg = min(
                    ys_n.numel(),
                    1024,
                )

                if ys_n.numel() > max_neg:
                    perm = torch.randperm(
                        ys_n.numel(),
                        device=device,
                    )[:max_neg]

                    ys_n = ys_n[perm]
                    xs_n = xs_n[perm]

                d_n = depth[
                    ys_n,
                    xs_n,
                ]

                x_cam_n = (xs_n.float() - cam.cx) / cam.fx * d_n

                y_cam_n = (ys_n.float() - cam.cy) / cam.fy * d_n

                z_cam_n = d_n

                ones_n = torch.ones_like(z_cam_n)

                p_cam_n = torch.stack(
                    [
                        x_cam_n,
                        y_cam_n,
                        z_cam_n,
                        ones_n,
                    ],
                    dim=-1,
                )

                Twc = cam.Twc.to(device)

                p_world_h_n = p_cam_n @ Twc

                p_world_n = p_world_h_n[..., :3] / p_world_h_n[..., 3:]

                dists_n = torch.cdist(
                    p_world_n,
                    means,
                )

                (
                    knn_dists_n,
                    knn_idx_n,
                ) = torch.topk(
                    dists_n,
                    k=min(
                        self.k_nn,
                        N,
                    ),
                    dim=-1,
                    largest=False,
                )

                del dists_n

                local_scales_n = scales[knn_idx_n]

                denom_n = local_scales_n.norm(dim=-1) + 1e-6

                d_local_n = knn_dists_n / denom_n

                valid_n = d_local_n < self.local_radius_thresh

                if valid_n.any():
                    d_local_valid_n = d_local_n.masked_fill(
                        ~valid_n,
                        1e9,
                    )

                    weights_n = torch.exp(-0.5 * d_local_valid_n**2)

                    weights_sum_n = (
                        weights_n.sum(
                            dim=-1,
                            keepdim=True,
                        )
                        + 1e-8
                    )

                    weights_n = weights_n / weights_sum_n

                    mask_vals_n = (
                        mask[
                            ys_n,
                            xs_n,
                        ]
                        .unsqueeze(-1)
                        .float()
                    )

                    contrib_n = (1.0 - mask_vals_n) * weights_n

                    flat_idx_n = knn_idx_n.reshape(-1)

                    flat_contrib_n = contrib_n.reshape(-1)

                    flat_valid_n = valid_n.reshape(-1)

                    flat_idx_n = flat_idx_n[flat_valid_n]

                    flat_contrib_n = flat_contrib_n[flat_valid_n]

                    neg_score.index_add_(
                        0,
                        flat_idx_n,
                        flat_contrib_n,
                    )

                    neg_votes.index_add_(
                        0,
                        flat_idx_n,
                        flat_contrib_n,
                    )

                    affected_n = torch.unique(flat_idx_n)

                    visible_views[affected_n] += 1

            # -----------------------------------------------------
            # Per-view lifting diagnostics
            # -----------------------------------------------------

            lifting_per_view.append(
                {
                    "view_id": int(view_id),
                    "view_name": str(view_name),
                    "skipped": False,
                    "skip_reason": None,
                    "positive_pixels": int(view_positive_pixels),
                    "sampled_positive_pixels": int(view_sampled_positive_pixels),
                    "knn_candidates": int(view_knn_candidates),
                    "local_radius_pass": int(view_local_radius_pass),
                    "local_radius_pass_ratio": (
                        float(view_local_radius_pass / view_knn_candidates)
                        if view_knn_candidates > 0
                        else 0.0
                    ),
                    "depth_consistency_pass": int(view_depth_consistency_pass),
                    "depth_consistency_pass_ratio": (
                        float(view_depth_consistency_pass / view_local_radius_pass)
                        if view_local_radius_pass > 0
                        else 0.0
                    ),
                }
            )

        # ---------------------------------------------------------
        # Alignment failure warning
        # ---------------------------------------------------------

        if skipped_views:
            logger.warning(
                "Depth alignment failed for {n}/{total} views (skipped).",
                n=skipped_views,
                total=len(cameras),
            )

        # ---------------------------------------------------------
        # Combine positive / negative evidence
        # ---------------------------------------------------------

        pos = self.lambda_seed * seed_score

        neg = self.lambda_neg * neg_score

        score_raw = pos / (pos + neg + 1e-8)

        # ---------------------------------------------------------
        # Multi-view consistency
        # ---------------------------------------------------------

        positive_ratio = positive_views.float() / (visible_views.float() + 1e-8)

        pass_visible = visible_views >= self.min_visible_views

        pass_positive = positive_views >= self.min_positive_views

        pass_seed = seed_views >= self.min_seed_views

        pass_positive_ratio = positive_ratio >= self.min_positive_ratio

        keep = pass_visible & pass_positive & pass_seed & pass_positive_ratio

        score = torch.where(
            keep,
            score_raw,
            torch.zeros_like(score_raw),
        )

        changed_gaussians = score > self.final_thresh

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        if self.diagnostics_enabled:
            self._sync_cuda()

            lifting_total_sec = time.perf_counter() - lift_start

            # Only use Gaussians with some visibility/evidence for
            # meaningful distribution statistics.
            observed_mask = visible_views > 0

            if observed_mask.any():
                visible_observed = visible_views[observed_mask].float()

                positive_observed = positive_views[observed_mask].float()

                positive_ratio_observed = positive_ratio[observed_mask]

                visible_mean = float(visible_observed.mean().item())

                visible_max = int(visible_observed.max().item())

                positive_mean = float(positive_observed.mean().item())

                positive_max = int(positive_observed.max().item())

                positive_ratio_mean = float(positive_ratio_observed.mean().item())

            else:
                visible_mean = 0.0
                visible_max = 0
                positive_mean = 0.0
                positive_max = 0
                positive_ratio_mean = 0.0

            # Score statistics.
            score_nonzero = score[score > 0]

            if score_nonzero.numel() > 0:
                score_stats = {
                    "nonzero_count": int(score_nonzero.numel()),
                    "mean": float(score_nonzero.mean().item()),
                    "min": float(score_nonzero.min().item()),
                    "p50": float(
                        torch.quantile(
                            score_nonzero,
                            0.50,
                        ).item()
                    ),
                    "p90": float(
                        torch.quantile(
                            score_nonzero,
                            0.90,
                        ).item()
                    ),
                    "p95": float(
                        torch.quantile(
                            score_nonzero,
                            0.95,
                        ).item()
                    ),
                    "max": float(score_nonzero.max().item()),
                }

            else:
                score_stats = {
                    "nonzero_count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                }

            # IMPORTANT:
            # These failure counts overlap because one Gaussian can fail
            # more than one multi-view condition. They must NOT be summed.
            multiview_stats = {
                "gaussians_observed": int(observed_mask.sum().item()),
                "gaussians_with_positive_evidence": int(len(gaussians_with_positive_evidence_set)),
                "visible_views_mean_observed": visible_mean,
                "visible_views_max": visible_max,
                "positive_views_mean_observed": positive_mean,
                "positive_views_max": positive_max,
                "positive_ratio_mean_observed": positive_ratio_mean,
                "failed_min_visible": int((observed_mask & (~pass_visible)).sum().item()),
                "failed_min_positive": int((observed_mask & (~pass_positive)).sum().item()),
                "failed_min_seed": int((observed_mask & (~pass_seed)).sum().item()),
                "failed_positive_ratio": int((observed_mask & (~pass_positive_ratio)).sum().item()),
                "passed_all_filters": int(keep.sum().item()),
                "passed_final_threshold": int(changed_gaussians.sum().item()),
            }

            self.last_stats = {
                # Existing summary
                "total_views": int(len(cameras)),
                "valid_views": int(len(cameras) - skipped_views),
                "skipped_views": int(skipped_views),
                # Timing
                "depth_estimation_sec": float(depth_estimation_sec),
                "depth_alignment_sec": float(depth_alignment_sec),
                "lifting_total_sec": float(lifting_total_sec),
                "lifting_other_sec": float(
                    max(
                        0.0,
                        lifting_total_sec - depth_estimation_sec - depth_alignment_sec,
                    )
                ),
                # Alignment details
                "depth_alignment_per_view": (depth_alignment_per_view),
                # Lifting details
                "lifting_per_view": (lifting_per_view),
                # Aggregate positive evidence
                "positive_pixels_total": int(total_positive_pixels),
                "positive_pixels_sampled": int(sampled_positive_pixels),
                # kNN candidates
                "knn_candidates": int(total_knn_candidates),
                # Local-radius gate
                "local_radius_pass": int(local_radius_pass),
                "local_radius_pass_ratio": (
                    float(local_radius_pass / total_knn_candidates)
                    if total_knn_candidates > 0
                    else 0.0
                ),
                # Depth-consistency gate
                "depth_consistency_pass": int(depth_consistency_pass),
                "depth_consistency_pass_ratio": (
                    float(depth_consistency_pass / local_radius_pass)
                    if local_radius_pass > 0
                    else 0.0
                ),
                # Multi-view
                "multiview": (multiview_stats),
                # Final score
                "score_stats": (score_stats),
            }

        return changed_gaussians
