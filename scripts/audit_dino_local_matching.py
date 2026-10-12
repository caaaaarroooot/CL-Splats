#!/usr/bin/env python3
"""CPU-only synthetic DINO patch matching audit. Not real DINO/GPU evidence."""
import argparse
import json
import numpy as np

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

def local_score(old, new, radius):
    h, w, d = old.shape
    pad = np.pad(old, ((radius, radius), (radius, radius), (0, 0)))
    score = np.full((h, w), -1.0, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = pad[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            score = np.maximum(score, (neighbor * new).sum(-1))
    return score

def audit(seed=42, size=24, dim=64, threshold=0.8):
    rng = np.random.default_rng(seed)
    old_unique = norm(rng.normal(size=(size, size, dim)).astype('float32'))
    tile = norm(rng.normal(size=(2, 2, dim)).astype('float32'))
    old_repeat = np.tile(tile, (size//2, size//2, 1))
    change = np.zeros((size, size), dtype=bool)
    c = size // 2
    change[c-3:c+3, c-3:c+3] = True
    valid = np.zeros_like(change)
    valid[2:-2, 2:-2] = True
    out = []
    for name, old in [('unique', old_unique), ('repetitive', old_repeat)]:
        for shift in (0, 1):
            new = old.copy()
            if shift:
                new[:, 1:] = old[:, :-1]
                new[:, 0] = norm(rng.normal(size=(size, dim)).astype('float32'))
            new[change] = norm(rng.normal(size=(change.sum(), dim)).astype('float32'))
            for radius in (0, 1, 2):
                detected = local_score(old, new, radius) < threshold
                tp = int((detected & change & valid).sum())
                fp = int((detected & ~change & valid).sum())
                fn = int((~detected & change & valid).sum())
                tn = int((~detected & ~change & valid).sum())
                out.append({'case': name, 'shift': shift, 'radius': radius,
                            'FPR': fp/(fp+tn), 'recall': tp/(tp+fn),
                            'precision': tp/(tp+fp) if tp+fp else None})
    new = old_unique.copy()
    new[change] = np.roll(old_unique, 1, axis=1)[change]
    for radius in (0, 1):
        detected = local_score(old_unique, new, radius) < threshold
        out.append({'case': 'neighbor-lookalike-change', 'shift': 0,
                    'radius': radius, 'FPR': float(detected[valid & ~change].mean()),
                    'recall': float(detected[change].mean())})
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--size', type=int, default=24)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--threshold', type=float, default=0.8)
    args = p.parse_args()
    print(json.dumps({'scope': 'synthetic_features_only', 'parameters': vars(args),
                      'results': audit(**vars(args))}, indent=2))
