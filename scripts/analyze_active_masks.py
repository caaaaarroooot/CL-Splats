from pathlib import Path

import torch


ROOT = Path("outputs/active_masks")
VIEWS = [5, 15, 30, 50]


data = {
    n: torch.load(
        ROOT / f"active_{n}.pt",
        map_location="cpu",
        weights_only=False,
    )
    for n in VIEWS
}


print()
print("=" * 80)
print("ACTIVE COUNTS")
print("=" * 80)

for n in VIEWS:
    print(
        f"{n:>2} views : "
        f"{data[n]['active_count']:>6} active Gaussians"
    )


print()
print("=" * 80)
print("PAIRWISE OVERLAP")
print("=" * 80)

for i, a in enumerate(VIEWS):
    for b in VIEWS[i + 1:]:

        A = data[a]["active_mask"].bool()
        B = data[b]["active_mask"].bool()

        intersection = int((A & B).sum())
        union = int((A | B).sum())

        only_a = int((A & ~B).sum())
        only_b = int((B & ~A).sum())

        recall_a_in_b = (
            intersection / int(A.sum())
            if int(A.sum()) > 0
            else 0.0
        )

        coverage_of_b_by_a = (
            intersection / int(B.sum())
            if int(B.sum()) > 0
            else 0.0
        )

        jaccard = (
            intersection / union
            if union > 0
            else 0.0
        )

        print()
        print(f"{a} vs {b}")
        print(f"  intersection       : {intersection}")
        print(f"  only {a:<2}            : {only_a}")
        print(f"  only {b:<2}            : {only_b}")
        print(
            f"  fraction of {a} retained in {b}: "
            f"{recall_a_in_b:.4%}"
        )
        print(
            f"  {b}-view coverage captured by {a}: "
            f"{coverage_of_b_by_a:.4%}"
        )
        print(
            f"  Jaccard IoU        : "
            f"{jaccard:.4%}"
        )


# ---------------------------------------------------------
# 50-view를 observation-rich reference로 본 coverage
# ---------------------------------------------------------

reference = data[50]["active_mask"].bool()

print()
print("=" * 80)
print("COVERAGE RELATIVE TO 50-VIEW REFERENCE")
print("=" * 80)

for n in [5, 15, 30]:

    mask = data[n]["active_mask"].bool()

    overlap = int(
        (mask & reference).sum()
    )

    reference_count = int(
        reference.sum()
    )

    missed = int(
        (reference & ~mask).sum()
    )

    extra = int(
        (mask & ~reference).sum()
    )

    coverage = (
        overlap / reference_count
        if reference_count > 0
        else 0.0
    )

    print()
    print(f"{n} views")
    print(
        f"  overlap with 50     : {overlap}"
    )
    print(
        f"  missed from 50      : {missed}"
    )
    print(
        f"  active only in {n:<2}    : {extra}"
    )
    print(
        f"  50-view coverage    : {coverage:.4%}"
    )


# ---------------------------------------------------------
# Incrementally discovered Gaussians
# ---------------------------------------------------------

A5 = data[5]["active_mask"].bool()
A15 = data[15]["active_mask"].bool()
A30 = data[30]["active_mask"].bool()
A50 = data[50]["active_mask"].bool()

print()
print("=" * 80)
print("INCREMENTAL DISCOVERY")
print("=" * 80)

print(
    "15 adds beyond 5 :",
    int((A15 & ~A5).sum()),
)

print(
    "30 adds beyond 15:",
    int((A30 & ~A15).sum()),
)

print(
    "50 adds beyond 30:",
    int((A50 & ~A30).sum()),
)

print(
    "50 active never seen by 5:",
    int((A50 & ~A5).sum()),
)
