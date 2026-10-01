#!/usr/bin/env bash
# Unattended CL-Splats experiment queue for 2026-10-01 night research.
# IMPORTANT: run only on branch research/night-test-20261001.
# This script does not modify source code; it only launches controlled experiments
# and copies each run's outputs to a unique directory so later runs do not overwrite them.

set -u -o pipefail

DATA_PATH="${DATA_PATH:-data/Blender-Levels/Level-1}"
BASE_PLY="${BASE_PLY:-outputs/gaussians_time_30000.ply}"
ROOT="${ROOT:-outputs/night_20261001}"
CHANGE_TYPE="${CHANGE_TYPE:-add}"
VIEWS="${VIEWS:-5}"

mkdir -p "$ROOT"/{logs,diag,ply,history}

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "research/night-test-20261001" ]]; then
  echo "ERROR: current branch is '$current_branch'."
  echo "Run this only on research/night-test-20261001."
  exit 2
fi

if [[ ! -f "$BASE_PLY" ]]; then
  echo "ERROR: pretrained PLY not found: $BASE_PLY"
  exit 2
fi

run_case() {
  local label="$1"
  local seed="$2"
  local iters="$3"
  shift 3

  local diag_dir="$ROOT/diag/$label"
  local log_file="$ROOT/logs/$label.log"
  mkdir -p "$diag_dir"

  echo "============================================================"
  echo "RUN: $label  seed=$seed  views=$VIEWS  iters=$iters"
  echo "START: $(date -Is)"
  echo "============================================================"

  cl-splats-train \
    --offline \
    --data-path "$DATA_PATH" \
    --change-type "$CHANGE_TYPE" \
    --white-background \
    --eval \
    model.pretrained_ply="$BASE_PLY" \
    train.num_change_views="$VIEWS" \
    train.view_sample_seed="$seed" \
    train.iters_per_timestep="$iters" \
    diagnostics.enabled=true \
    diagnostics.out_dir="$diag_dir" \
    history.log_history=true \
    wandb_mode=disabled \
    "$@" 2>&1 | tee "$log_file"

  local exit_code=${PIPESTATUS[0]}
  if [[ $exit_code -ne 0 ]]; then
    echo "FAILED: $label (exit=$exit_code)" | tee -a "$ROOT/FAILED.txt"
    return 0
  fi

  local src_ply="outputs/ply/ply_${VIEWS}_${iters}.ply"
  local src_hist="outputs/ply/history/t0001.pt"

  if [[ -f "$src_ply" ]]; then
    cp "$src_ply" "$ROOT/ply/${label}.ply"
  else
    echo "WARNING: missing PLY for $label: $src_ply" | tee -a "$ROOT/WARNINGS.txt"
  fi

  if [[ -f "$src_hist" ]]; then
    cp "$src_hist" "$ROOT/history/${label}.pt"
  else
    echo "WARNING: missing history for $label: $src_hist" | tee -a "$ROOT/WARNINGS.txt"
  fi

  echo "END: $(date -Is)"
  echo
}

# -----------------------------------------------------------------------------
# Part A. Convergence attribution at the fixed baseline subset (seed=42).
# Densification is disabled by moving the refine window far beyond the run.
# These cases isolate which parameter groups are responsible for improvement.
# -----------------------------------------------------------------------------

for iters in 1000 3000; do
  # Full current pipeline: same-code baseline for clean comparison.
  run_case "conv_full_s42_i${iters}" 42 "$iters"

  # No densification: all Gaussian parameters may optimize, but no split/duplicate growth.
  run_case "conv_nodens_s42_i${iters}" 42 "$iters" \
    train.densify_from_iter=1000000 \
    train.densify_until_iter=1000001

  # Appearance only: SH + opacity update. XYZ/scale/rotation fixed; no densification.
  run_case "conv_appearance_s42_i${iters}" 42 "$iters" \
    train.position_lr_init=0.0 \
    train.position_lr_final=0.0 \
    train.scaling_lr=0.0 \
    train.rotation_lr=0.0 \
    train.densify_from_iter=1000000 \
    train.densify_until_iter=1000001

  # Geometry only: xyz + scale + rotation update. SH/color + opacity fixed; no densification.
  run_case "conv_geometry_s42_i${iters}" 42 "$iters" \
    train.feature_lr=0.0 \
    train.opacity_lr=0.0 \
    train.densify_from_iter=1000000 \
    train.densify_until_iter=1000001

  # XYZ only: isolates position correction.
  run_case "conv_xyz_s42_i${iters}" 42 "$iters" \
    train.feature_lr=0.0 \
    train.opacity_lr=0.0 \
    train.scaling_lr=0.0 \
    train.rotation_lr=0.0 \
    train.densify_from_iter=1000000 \
    train.densify_until_iter=1000001

done

# -----------------------------------------------------------------------------
# Part B. Multi-seed robustness.
# The current 5-view evidence is based heavily on seed=42. This sweep tests whether
# low coverage / convergence behaviour is structural or an unlucky view subset.
# 1000 iters: broad seed sweep. 3000 iters: smaller confirmation sweep.
# -----------------------------------------------------------------------------

for seed in {0..19} 42; do
  run_case "seed5_s${seed}_i1000" "$seed" 1000
done

for seed in {0..9} 42; do
  run_case "seed5_s${seed}_i3000" "$seed" 3000
done

# -----------------------------------------------------------------------------
# Summary index for quick inspection after the queue finishes.
# -----------------------------------------------------------------------------

{
  echo "Finished: $(date -Is)"
  echo "Branch: $(git branch --show-current)"
  echo "Data: $DATA_PATH"
  echo "Change type: $CHANGE_TYPE"
  echo "Views: $VIEWS"
  echo "Base PLY: $BASE_PLY"
  echo
  echo "Diagnostics:"
  find "$ROOT/diag" -type f -name '*.json' | sort
  echo
  echo "PLYs:"
  find "$ROOT/ply" -type f -name '*.ply' | sort
} > "$ROOT/INDEX.txt"

echo "ALL QUEUED EXPERIMENTS FINISHED."
echo "Results: $ROOT"
