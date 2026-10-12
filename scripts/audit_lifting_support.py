#!/usr/bin/env python3
"""CPU-only synthetic stress tests of CL-Splats' existing-Gaussian lifting gates.

Does NOT reconstruct 3DGS or benchmark Spark. Intended to distinguish structural
absence of old-Gaussian support from threshold / anisotropy / top-k effects.
No third-party dependencies. Run: python scripts/audit_lifting_support.py
"""
import argparse
import json
import math
import random


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def dist(a, b):
    return norm([x - y for x, y in zip(a, b)])


def assess(query, means, scales, k, radius, abs_tol, rel_tol):
    ranked = sorted(range(len(means)), key=lambda i: dist(query, means[i]))[:k]
    radius_pass = []
    depth_pass = []
    for i in ranked:
        radius_ok = dist(query, means[i]) / (norm(scales[i]) + 1e-6) < radius
        depth_ok = abs(query[2] - means[i][2]) < abs_tol + rel_tol * query[2]
        radius_pass.append(radius_ok)
        depth_pass.append(depth_ok)
    return {
        "radius_any": any(radius_pass),
        "depth_any": any(depth_pass),
        "both_any": any(a and b for a, b in zip(radius_pass, depth_pass)),
    }


def plane_test(args):
    rng = random.Random(args.seed)
    grid = [(-1.0 + i * 0.1) for i in range(21)]
    means = [(x, y, args.old_z) for x in grid for y in grid]
    scales = [(args.scale, args.scale, args.scale_z)] * len(means)
    rows = []
    for new_z in args.depths:
        counters = {"radius_any": 0, "depth_any": 0, "both_any": 0}
        for _ in range(args.samples):
            p = (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), new_z)
            result = assess(p, means, scales, args.k, args.radius, args.abs_tol, args.rel_tol)
            for key in counters:
                counters[key] += int(result[key])
        rows.append({
            "new_surface_z": new_z,
            "depth_gap": round(abs(args.old_z - new_z), 5),
            "depth_gate_tolerance": round(args.abs_tol + args.rel_tol * new_z, 5),
            **{key + "_rate": round(value / args.samples, 4) for key, value in counters.items()},
        })
    return rows


def anisotropy_test():
    # One elongated Gaussian is outside Euclidean top-8 but closest under
    # its anisotropic covariance; eight small distractors dominate Euclidean kNN.
    query = (0.8, 0.0, 5.0)
    means = [(0.0, 0.0, 5.0)]
    scales = [(1.2, 0.03, 0.03)]
    for j in range(8):
        theta = 2 * math.pi * j / 8
        means.append((0.8 + 0.1 * math.cos(theta), 0.1 * math.sin(theta), 5.0))
        scales.append((0.01, 0.01, 0.01))
    euclidean_order = sorted(range(len(means)), key=lambda i: dist(query, means[i]))
    # Axis-aligned example only; real covariance must include quaternion rotation.
    mahalanobis = [norm([(query[j] - mu[j]) / s[j] for j in range(3)])
                   for mu, s in zip(means, scales)]
    return {
        "query": query,
        "euclidean_top8_indices": euclidean_order[:8],
        "elongated_gaussian_index": 0,
        "elongated_gaussian_in_top8": 0 in euclidean_order[:8],
        "elongated_gaussian_mahalanobis": round(mahalanobis[0], 4),
        "best_mahalanobis_index": min(range(len(means)), key=lambda i: mahalanobis[i]),
        "nearest_compact_mahalanobis": round(min(mahalanobis[1:]), 4),
        "warning": "Synthetic counterexample only; real-data prevalence unmeasured.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-z", type=float, default=5.0)
    p.add_argument("--depths", nargs="+", type=float, default=[5.0, 4.95, 4.8, 4.0, 3.0])
    p.add_argument("--scale", type=float, default=0.06)
    p.add_argument("--scale-z", type=float, default=0.025)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--radius", type=float, default=2.5)
    p.add_argument("--abs-tol", type=float, default=0.05)
    p.add_argument("--rel-tol", type=float, default=0.05)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.k < 1 or args.samples < 1 or args.old_z <= 0 or min(args.depths) <= 0:
        p.error("k, samples and all depths must be positive")
    print(json.dumps({
        "type": "synthetic_cpu_only_not_a_real_3dgs_experiment",
        "parameters": vars(args),
        "plane_support": plane_test(args),
        "anisotropic_topk_counterexample": anisotropy_test(),
        "interpretation": "For a newly inserted surface far from all old Gaussian centers, both old-center radius and depth-consistency gates reject positive evidence. Loosening only one gate cannot repair the other. New-Gaussian seeding or separate novel-space evidence is needed for such cases. Covariance-aware reranking after Euclidean top-k cannot recover candidates absent from top-k."
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
