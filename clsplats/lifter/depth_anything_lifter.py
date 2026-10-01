"""Depth-Anything V2 lifter.

Estimates monocular depth using Depth-Anything V2 from HuggingFace and
lifts 2-D change masks into a per-Gaussian change mask using multi-view
depth back-projection and Gaussian proximity scoring.
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


def _align_relative_depth_with_stats(
    mono: torch.Tensor,
    rendered_depth: torch.Tensor,
    change_mask: torch.Tensor,
    min_pixels: int = 200,
) -> tuple[Optional[torch.Tensor], dict]:
    """Align relative monocular depth and return diagnostic statistics.

    This helper preserves the original alignment algorithm while exposing
    diagnostic information about alignment quality and failure reasons.
    """
    valid = (
        (~change_mask)
        & (rendered_depth > 1e-6)
        & torch.isfinite(rendered_depth)
        & torch.isfinite(mono)
    )

    anchor_pixels = int(valid.sum().item())

    stats = {
        "success": False,
        "failure_reason": None,
        "anchor_pixels": anchor_pixels,
        "r2": None,
        "scale_a": None,
        "offset_b": None,
    }

    if anchor_pixels < min_pixels:
        stats["failure_reason"] = "insufficient_anchor_pixels"
        return None, stats

    x = mono[valid].float()
    y = 1.0 / rendered_depth[valid].float()

    ones = torch.ones_like(x)
    A = torch.stack([x, ones], dim=-1)

    solution = torch.linalg.lstsq(
        A,
        y.unsqueeze(-1),
    ).solution.squeeze(-1)

    a = float(solution[0].item())
    b = float(solution[1].item())

    stats["scale_a"] = a
    stats["offset_b"] = b

    prediction = a * x + b

    ss_res = ((prediction - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)

    r2 = float((1.0 - ss_res / ss_tot).item())
    stats["r2"] = r2

    if r2 < 0.3:
        stats["failure_reason"] = "low_r2"
        return None, stats

    disparity = (a * mono + b).clamp_min(1e-6)
    aligned_depth = 1.0 / disparity

    stats["success"] = True

    return aligned_depth, stats


def align_relative_depth(
    mono: torch.Tensor,
    rendered_depth: torch.Tensor,
    change_mask: torch.Tensor,
    min_pixels: int = 200,
) -> Optional[torch.Tensor]:
    """Anchor relative monocular depth to the scene's metric scale.

    This public wrapper preserves the original API used by existing code
    and tests. Diagnostic callers should use
    ``_align_relative_depth_with_stats`` instead.
    """
    aligned_depth, _ = _align_relative_depth_with_stats(
        mono=mono,
        rendered_depth=rendered_depth,
        change_mask=change_mask,
        min_pixels=min_pixels,
    )

    return aligned_depth


class DepthAnythingLifter(BaseLifter):
    """Lifter that uses Depth-Anything V2 for monocular depth estimation."""

    def __init__(self, cfg: CLSplatsConfig):
        super().__init__(cfg)
        from transformers import pipeline as hf_pipeline

        model_name = cfg.lifter.depth_model
        self._pipe = hf_pipeline(task="depth-estimation", model=model_name)

        # Lifting hyper-parameters — read directly from typed config
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
        self.max_positive_pixels = lcfg.max_positive_pixels
        self.positive_chunk_size = lcfg.positive_chunk_size
        if self.max_positive_pixels <= 0:
            raise ValueError("lifter.max_positive_pixels must be > 0")
        if self.positive_chunk_size <= 0:
            raise ValueError("lifter.positive_chunk_size must be > 0")
        self.diagnostics_enabled = cfg.diagnostics.enabled
        self.last_stats: dict[str, object] = {}

    def _sync_cuda(self) -> None:
        if self.diagnostics_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def estimate_depth(self, observation: Image) -> torch.Tensor:
        """Estimate depth from an observation image ``[H, W, 3]`` in ``[0, 1]``.

        Returns a depth map as a ``float32`` tensor ``[H, W]`` normalised to
        ``[0, 1]``.
        """
        obs_np = (observation.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        pil_img = PILImage.fromarray(obs_np)

        depth_output = self._pipe(pil_img)["depth"]  # PIL Image

        # Bug 2 fix: convert PIL Image → numpy → tensor properly
        depth_np = np.array(depth_output, dtype=np.float32)
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
        """Multi-view lifting.

        For each view, estimates depth with Depth-Anything, back-projects
        changed pixels to 3-D, assigns evidence to nearby Gaussians, and
        accumulates multi-view positive/negative evidence.

        Returns:
            Boolean mask ``(N,)`` indicating which Gaussians are changed.
        """
        device = gaussians.params.positions.device
        N = gaussians.params.positions.shape[0]
        self.last_stats = {}

        depth_estimation_sec = 0.0
        depth_alignment_sec = 0.0

        if self.diagnostics_enabled:
            self._sync_cuda()
            lift_start = time.perf_counter()
        else:
            lift_start = 0.0

        seed_score = torch.zeros(N, device=device)
        seed_votes = torch.zeros(N, device=device)
        neg_score = torch.zeros(N, device=device)
        neg_votes = torch.zeros(N, device=device)
        visible_views = torch.zeros(N, dtype=torch.int32, device=device)
        positive_views = torch.zeros(N, dtype=torch.int32, device=device)
        seed_views = torch.zeros(N, dtype=torch.int32, device=device)

        means = gaussians.params.positions  # (N, 3)
        scales = gaussians.params.scales  # (N, 3)

        skipped_views = 0
        alignment_per_view = []
        lifting_per_view = []

        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            positive_pixels_total = 0
            positive_pixels_sampled = 0
            positive_chunks = 0
            radius_candidates = 0
            radius_passed = 0
            depth_passed = 0
            obs = cam.original_image.permute(1, 2, 0).contiguous()

            if self.diagnostics_enabled:
                self._sync_cuda()
                depth_start = time.perf_counter()

            depth = self.estimate_depth(obs).to(device)

            if self.diagnostics_enabled:
                self._sync_cuda()
                depth_estimation_sec += time.perf_counter() - depth_start

            mask = mask.to(device)

            # Monocular depth is relative — anchor it to the scene's metric
            # scale using the rendered depth of the current model. Without
            # the alignment, back-projections land in empty space and the
            # depth-consistency gate rejects every Gaussian.
            if rendered_depths is not None:
                if self.diagnostics_enabled:
                    self._sync_cuda()
                    alignment_start = time.perf_counter()

                aligned, alignment_stats = _align_relative_depth_with_stats(
                    mono=depth,
                    rendered_depth=rendered_depths[view_id].to(device),
                    change_mask=mask > 0.5,
                )

                if self.diagnostics_enabled:
                    self._sync_cuda()
                    depth_alignment_sec += time.perf_counter() - alignment_start

                    alignment_per_view.append(
                        {
                            "view_id": view_id,
                            "image_name": cam.image_name,
                            **alignment_stats,
                        }
                    )

                if aligned is None:
                    skipped_views += 1
                    continue

                depth = aligned

            # --- Positive pixels (changed) ---
            pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)
            positive_pixels_total = int(pos_pixels.sum().item())

            if pos_pixels.any():
                ys, xs = torch.nonzero(pos_pixels, as_tuple=True)

                # Sample up to the configured per-view budget once, then process
                # that sample in bounded chunks. This keeps peak cdist memory
                # near the original 2048-pixel implementation while allowing a
                # larger total evidence budget per view.
                max_pos = min(ys.numel(), self.max_positive_pixels)
                if ys.numel() > max_pos:
                    perm = torch.randperm(ys.numel(), device=device)[:max_pos]
                    ys = ys[perm]
                    xs = xs[perm]

                positive_pixels_sampled = int(ys.numel())
                positive_chunks = (
                    (positive_pixels_sampled + self.positive_chunk_size - 1)
                    // self.positive_chunk_size
                )

                # A Gaussian may be reached by multiple chunks from the SAME
                # camera. Multi-view counters must therefore be incremented
                # only once per camera, after unioning all chunk hits.
                positive_affected_this_view = torch.zeros(
                    N, dtype=torch.bool, device=device
                )

                Twc = cam.Twc.to(device)  # [4, 4]
                Tcw = torch.inverse(Twc)  # [4, 4], transposed convention

                for chunk_start in range(0, positive_pixels_sampled, self.positive_chunk_size):
                    chunk_end = min(
                        chunk_start + self.positive_chunk_size,
                        positive_pixels_sampled,
                    )
                    ys_c = ys[chunk_start:chunk_end]
                    xs_c = xs[chunk_start:chunk_end]
                    d = depth[ys_c, xs_c]

                    # Back-project to camera coordinates
                    x_cam = (xs_c.float() - cam.cx) / cam.fx * d
                    y_cam = (ys_c.float() - cam.cy) / cam.fy * d
                    z_cam = d

                    ones = torch.ones_like(z_cam)
                    p_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)
                    p_world_h = p_cam @ Twc
                    p_world = p_world_h[..., :3] / p_world_h[..., 3:]

                    # kNN in Gaussian means. Chunking bounds the M dimension so
                    # torch.cdist peak memory does not scale with total budget.
                    dists = torch.cdist(p_world, means)
                    knn_dists, knn_idx = torch.topk(
                        dists,
                        k=min(self.k_nn, N),
                        dim=-1,
                        largest=False,
                    )
                    del dists

                    local_scales = scales[knn_idx]
                    denom = local_scales.norm(dim=-1) + 1e-6
                    d_local = knn_dists / denom

                    valid = d_local < self.local_radius_thresh
                    radius_candidates += int(valid.numel())
                    radius_passed += int(valid.sum().item())

                    if not valid.any():
                        continue

                    # Depth consistency: project only k neighbour means.
                    knn_means = means[knn_idx]
                    M, k = knn_means.shape[:2]
                    knn_means_h = torch.cat(
                        [knn_means, torch.ones(M, k, 1, device=device)], dim=-1
                    )
                    knn_cam = knn_means_h @ Tcw
                    z_knn = knn_cam[..., 2]

                    depth_pix = d.unsqueeze(-1)
                    depth_ok = (z_knn - depth_pix).abs() < (
                        self.depth_tol_abs + self.depth_tol_rel * depth_pix
                    )
                    valid_final = valid & depth_ok
                    depth_passed += int(valid_final.sum().item())

                    if not valid_final.any():
                        continue

                    d_local_valid = d_local.masked_fill(~valid_final, 1e9)
                    weights = torch.exp(-0.5 * d_local_valid**2)
                    weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-8
                    weights = weights / weights_sum

                    mask_vals = mask[ys_c, xs_c].unsqueeze(-1).float()
                    contrib = mask_vals * weights

                    flat_idx = knn_idx.view(-1)
                    flat_contrib = contrib.view(-1)
                    flat_valid = valid_final.view(-1)

                    flat_idx = flat_idx[flat_valid]
                    flat_contrib = flat_contrib[flat_valid]

                    seed_score.index_add_(0, flat_idx, flat_contrib)
                    seed_votes.index_add_(0, flat_idx, flat_contrib)

                    affected = torch.unique(flat_idx)
                    positive_affected_this_view[affected] = True

                affected = torch.nonzero(
                    positive_affected_this_view, as_tuple=False
                ).squeeze(1)
                if affected.numel() > 0:
                    positive_views[affected] += 1
                    seed_views[affected] += 1
                    visible_views[affected] += 1

            # --- Weak negatives from un-masked pixels (sub-sampled) ---
            neg_pixels = (~pos_pixels) & torch.isfinite(depth) & (depth > 0)
            if neg_pixels.any():
                ys_n, xs_n = torch.nonzero(neg_pixels, as_tuple=True)
                max_neg = min(ys_n.numel(), 1024)
                perm = torch.randperm(ys_n.numel(), device=device)[:max_neg]
                ys_n = ys_n[perm]
                xs_n = xs_n[perm]
                d_n = depth[ys_n, xs_n]

                x_cam_n = (xs_n.float() - cam.cx) / cam.fx * d_n
                y_cam_n = (ys_n.float() - cam.cy) / cam.fy * d_n
                z_cam_n = d_n

                ones_n = torch.ones_like(z_cam_n)
                p_cam_n = torch.stack([x_cam_n, y_cam_n, z_cam_n, ones_n], dim=-1)
                Twc = cam.Twc.to(device)
                p_world_h_n = p_cam_n @ Twc
                # Bug 4 fix: removed incorrect .unsqueeze(-1)
                p_world_n = p_world_h_n[..., :3] / p_world_h_n[..., 3:]

                dists_n = torch.cdist(p_world_n, means)
                knn_dists_n, knn_idx_n = torch.topk(
                    dists_n, k=min(self.k_nn, N), dim=-1, largest=False
                )

                local_scales_n = scales[knn_idx_n]
                denom_n = local_scales_n.norm(dim=-1) + 1e-6
                d_local_n = knn_dists_n / denom_n

                valid_n = d_local_n < self.local_radius_thresh
                if valid_n.any():
                    d_local_valid_n = d_local_n.masked_fill(~valid_n, 1e9)
                    weights_n = torch.exp(-0.5 * d_local_valid_n**2)
                    weights_sum_n = weights_n.sum(dim=-1, keepdim=True) + 1e-8
                    weights_n = weights_n / weights_sum_n

                    mask_vals_n = mask[ys_n, xs_n].unsqueeze(-1).float()
                    contrib_n = (1.0 - mask_vals_n) * weights_n

                    flat_idx_n = knn_idx_n.view(-1)
                    flat_contrib_n = contrib_n.view(-1)
                    flat_valid_n = valid_n.view(-1)

                    flat_idx_n = flat_idx_n[flat_valid_n]
                    flat_contrib_n = flat_contrib_n[flat_valid_n]

                    neg_score.index_add_(0, flat_idx_n, flat_contrib_n)
                    neg_votes.index_add_(0, flat_idx_n, flat_contrib_n)

                    affected_n = torch.unique(flat_idx_n)
                    visible_views[affected_n] += 1

            lifting_per_view.append(
                {
                    "view_id": view_id,
                    "image_name": cam.image_name,
                    "positive_pixels_total": positive_pixels_total,
                    "positive_pixels_sampled": positive_pixels_sampled,
                    "positive_chunks": positive_chunks,
                    "radius_candidates": radius_candidates,
                    "radius_passed": radius_passed,
                    "depth_passed": depth_passed,
                }
            )

        if skipped_views:
            logger.warning(
                "Depth alignment failed for {n}/{total} views (skipped).",
                n=skipped_views,
                total=len(cameras),
            )

        # Combine evidence
        # Bug 10 fix: confidence thresholding variables were computed but unused; removed.
        pos = self.lambda_seed * seed_score
        neg = self.lambda_neg * neg_score

        score = pos / (pos + neg + 1e-8)

        # Multi-view consistency filtering
        visible_ok = visible_views >= self.min_visible_views
        positive_ok = positive_views >= self.min_positive_views
        seed_ok = seed_views >= self.min_seed_views

        positive_ratio = positive_views.float() / (visible_views.float() + 1e-8)
        ratio_ok = positive_ratio >= self.min_positive_ratio

        keep = visible_ok & positive_ok & seed_ok & ratio_ok

        visible_passed = int(visible_ok.sum().item())
        visible_failed = int((~visible_ok).sum().item())

        positive_passed = int(positive_ok.sum().item())
        positive_failed = int((~positive_ok).sum().item())

        seed_passed = int(seed_ok.sum().item())
        seed_failed = int((~seed_ok).sum().item())

        ratio_passed = int(ratio_ok.sum().item())
        ratio_failed = int((~ratio_ok).sum().item())

        multiview_passed = int(keep.sum().item())
        multiview_failed = int((~keep).sum().item())
        score = torch.where(keep, score, torch.zeros_like(score))

        changed_gaussians = score > self.final_thresh

        final_score_passed = int(changed_gaussians.sum().item())
        final_score_failed = int(multiview_passed - final_score_passed)

        if self.diagnostics_enabled:
            self._sync_cuda()

            lifting_total_sec = time.perf_counter() - lift_start

            self.last_stats = {
                "total_views": len(cameras),
                "valid_views": len(cameras) - skipped_views,
                "skipped_views": skipped_views,
                "max_positive_pixels": self.max_positive_pixels,
                "positive_chunk_size": self.positive_chunk_size,
                "depth_estimation_sec": depth_estimation_sec,
                "depth_alignment_sec": depth_alignment_sec,
                "lifting_total_sec": lifting_total_sec,
                "lifting_other_sec": max(
                    0.0,
                    lifting_total_sec - depth_estimation_sec - depth_alignment_sec,
                ),
                "depth_alignment_per_view": alignment_per_view,
                "lifting_per_view": lifting_per_view,
                "multiview_filter": {
                    "total_gaussians": N,
                    "visible_passed": visible_passed,
                    "visible_failed": visible_failed,
                    "positive_passed": positive_passed,
                    "positive_failed": positive_failed,
                    "seed_passed": seed_passed,
                    "seed_failed": seed_failed,
                    "ratio_passed": ratio_passed,
                    "ratio_failed": ratio_failed,
                    "multiview_passed": multiview_passed,
                    "multiview_failed": multiview_failed,
                },
                "final_score_filter": {
                    "threshold": self.final_thresh,
                    "passed": final_score_passed,
                    "failed": final_score_failed,
                },
            }

        return changed_gaussians
