"""Compare sparse-view active masks against a reference active mask.

Typical use:
    python tools/coverage_mask_metrics.py \
        --reference outputs/active_masks/active_50.pt \
        outputs/active_masks/active_5.pt \
        outputs/active_masks/active_15.pt

The reference mask is treated as a pseudo-ground-truth coverage target.  The
script reports how much of that reference is recovered by each candidate mask
and how many candidate Gaussians fall outside the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


_MASK_KEYS = (
    "active_mask",
    "mask",
    "changed_gaussians",
    "active_gaussians",
    "active",
)


def _extract_mask(obj: Any, path: Path) -> torch.Tensor:
    """Return a flattened boolean mask from common torch-save layouts."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().bool().flatten()

    if isinstance(obj, dict):
        for key in _MASK_KEYS:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value.detach().cpu().bool().flatten()

        tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0].detach().cpu().bool().flatten()

    raise ValueError(
        f"Could not find an active mask tensor in {path}. "
        f"Expected a tensor or a dict containing one of: {_MASK_KEYS}."
    )


def load_mask(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return _extract_mask(obj, path)


def compare(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    if candidate.numel() != reference.numel():
        raise ValueError(
            "Mask lengths differ: "
            f"candidate={candidate.numel()}, reference={reference.numel()}"
        )

    candidate = candidate.bool()
    reference = reference.bool()

    tp = int((candidate & reference).sum().item())
    fp = int((candidate & ~reference).sum().item())
    fn = int((~candidate & reference).sum().item())
    candidate_count = int(candidate.sum().item())
    reference_count = int(reference.sum().item())
    union = int((candidate | reference).sum().item())

    precision = tp / candidate_count if candidate_count else 0.0
    recall = tp / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    iou = tp / union if union else 0.0

    return {
        "candidate": candidate_count,
        "reference": reference_count,
        "intersection": tp,
        "candidate_only": fp,
        "missed_reference": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate active masks with a reference mask."
    )
    parser.add_argument(
        "candidates",
        nargs="+",
        type=Path,
        help="Candidate .pt active-mask files.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference .pt active-mask file, e.g. the 50-view mask.",
    )
    args = parser.parse_args()

    reference = load_mask(args.reference)

    print(f"Reference: {args.reference}  active={int(reference.sum().item())}")
    print(
        f"{'candidate':<34} {'active':>8} {'inter':>8} {'only':>8} "
        f"{'missed':>8} {'prec':>8} {'recall':>8} {'F1':>8} {'IoU':>8}"
    )

    for path in args.candidates:
        candidate = load_mask(path)
        metrics = compare(candidate, reference)
        print(
            f"{path.name:<34} "
            f"{metrics['candidate']:>8d} "
            f"{metrics['intersection']:>8d} "
            f"{metrics['candidate_only']:>8d} "
            f"{metrics['missed_reference']:>8d} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['iou']:>8.4f}"
        )


if __name__ == "__main__":
    main()
