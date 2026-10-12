# Depth calibration and seed insertion ablations

Status: research design only; no Spark/GPU execution.

## Stage A: isolate calibration
Freeze 5 RGB images, camera poses, DINO masks, old 3DGS depth, Depth Anything Small weights and seed. Compare D0 current uint8 PIL depth + least squares, D1 float predicted_depth + least squares, D2 float + Mastering Regional Eq.6 median/mean-absolute-deviation fit, D3 float + Huber IRLS, D4 reciprocal float feature + fit. Use spatially separated unchanged anchors for fit/validation. Track per-view valid anchors, R2, held-out depth/disparity residuals, positive-disparity fraction, valid views, and latency. Do not mistake old rendered depth for changed-object GT.

Eq.6: a=s(delta_old)/s(delta_mono); b=median(delta_old)-a*median(delta_mono); s(x)=mean(abs(x-median(x))); depth=1/(a*delta_mono+b). The paper uses *mean* absolute deviation around the median, not median absolute deviation. Check output polarity before forcing a positive scale. Reference: https://link.springer.com/article/10.1007/s11263-026-02941-w

## Stage B: isolate representation capacity versus geometric prior
At 100 steps with growth disabled, compare S0 existing Gaussian-only; S1 equal-count random/old-depth new seeds; S2 robustly calibrated new-depth seeds; S3 Mastering Eq.7 old/new depth ratio seeds; S4 two-view triangulated changed-pixel seeds inspired by EDGS. Fix number of seeds, colors/scales/opacity, view subset, held-out evaluation and optimization settings. Reprojection- and occlusion-filter seeds in every method. Require >=2 independent supporting views when possible; log rejected seeds. Report held-out changed-region PSNR/SSIM/LPIPS, synthetic GT geometry, unchanged-region preservation, births/prunes, and full latency.

Mastering Eq.7 is tailored to edited/unedited paired images; a physical ADD with newly occluded background can violate its assumptions. EDGS uses dense correspondence triangulation and reports 10-20s reconstruction plus 5-10s COLMAP in its official README; full EDGS is not an <=10s E2E solution. EDGS repo root LICENSE was not found on 2026-10-12, so do not infer commercial code permission. Sources: https://github.com/CompVis/EDGS and https://openaccess.thecvf.com/content/CVPR2026/html/Kotovenko_EDGS_Eliminating_Densification_for_Efficient_Convergence_of_3DGS_CVPR_2026_paper.html

Hypothesis: S2 improves local geometry and held-out image quality over S0 *and* equal-capacity S1 within 10 seconds. Reject if gains are absent at matched false-positive rate/time, or if unchanged preservation degrades. D2/D3 should be compared against D1 before crediting robust calibration; R2 alone can be misleading for low-variance anchor depth. This is a proposed experiment, not a completed test.
