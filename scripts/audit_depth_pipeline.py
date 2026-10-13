#!/usr/bin/env python3
"""Non-mutating audit of CL-Splats' Depth Anything HF pipeline path.

Usage:
  python scripts/audit_depth_pipeline.py --image /path/to/change_view.png --repeat 3
  python scripts/audit_depth_pipeline.py --image /path/to/change_view.png --device cpu

This script DOES NOT train or change the scene. It compares the float prediction
(predicted_depth) against the visualization PIL image (depth) currently
consumed by clsplats/lifter/depth_anything_lifter.py, and reports device/timing.
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def sync(actual_device):
    if str(actual_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "production-default"], default="auto")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    if args.repeat < 1:
        p.error("--repeat must be >=1")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "production-default":
        device = None
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")

    img = Image.open(args.image).convert("RGB")
    t0 = time.perf_counter()
    # Production lifter constructs pipeline without device=; test that exact default separately.
    if device is None:
        pipe = pipeline("depth-estimation", model=args.model)
    else:
        pipe = pipeline("depth-estimation", model=args.model, device=device)
    actual_device = str(pipe.device)
    model_param_device = str(next(pipe.model.parameters()).device)
    sync(actual_device)
    model_load_s = time.perf_counter() - t0

    times = []
    result = None
    for _ in range(args.repeat + 1):  # first inference is warm-up
        sync(actual_device)
        start = time.perf_counter()
        result = pipe(img)
        sync(actual_device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    raw = result["predicted_depth"].detach().float().cpu().numpy()
    raw = np.squeeze(raw)
    pil = np.asarray(result["depth"], dtype=np.float32) / 255.0
    if raw.shape != pil.shape:
        raise ValueError(f"Shape mismatch: raw {raw.shape} vs PIL {pil.shape}")
    span = float(raw.max() - raw.min())
    normalized = (raw - raw.min()) / max(span, 1e-12)
    err = pil - normalized

    payload = {
        "image": args.image,
        "model": args.model,
        "device_requested": args.device,
        "device_argument_passed": device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "model_parameter_device": model_param_device,
        "pipeline_device": actual_device,
        "image_shape_hw": list(raw.shape),
        "load_seconds": round(model_load_s, 4),
        "warmup_seconds": round(times[0], 4),
        "inference_seconds": [round(x, 4) for x in times[1:]],
        "inference_median_seconds": round(float(np.median(times[1:])), 4),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_unique_count": int(np.unique(raw).size),
        "visualization_unique_count": int(np.unique(pil).size),
        "visualization_dtype": str(np.asarray(result["depth"]).dtype),
        "normalized_raw_vs_visualization_mae": float(np.mean(np.abs(err))),
        "normalized_raw_vs_visualization_max_abs_error": float(np.max(np.abs(err))),
        "note": "The production lifter currently uses result['depth'] (8-bit PIL visualization), not result['predicted_depth'] (float). The normalized float prediction is the non-quantized comparator, not ground-truth depth.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
