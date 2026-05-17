# Number audit report

| Source | Field | Status | Manuscript value | JSON reference |
| --- | --- | --- | --- | --- |
| Zernike fit | ray_rms_mm | OK | 0.0006589174390985027 | expected ~0.00065 |
| Reprojection | left_rms_px | OK | 0.5137985684346419 | summary.json |
| Reprojection | right_rms_px | OK | 0.42159335928841274 | summary.json |
| Descriptors | baseline_mm | OK | 24.884604246022516 | summary.json |
| Descriptors | working_distance_mm | OK | 64.70220034514719 | summary.json |
| Descriptors | f_obj_mm | OK | 62.208914898071626 | summary.json |
| Descriptors | convergence_angle_deg | OK | 22.551396758555008 | summary.json |
| 26p model | px_rms | OK | 1.056459322023649 | expected 1.06 |
| 26p model | px_p50 | OK | 0.8718477289533714 | expected 0.87 |
| 26p model | px_p95 | OK | 1.8436965936908176 | expected 1.84 |
| Telecentric L0 | px_rms | OK | 14.552113564559823 | expected ~14.6 |
| Cross-val | rms_mean | OK | 1.069887588180309 | expected 1.07 |
| Cross-val | rms_std | OK | 0.017082326460859703 | |
| Bootstrap | b_95ci | OK | [23.475303081304087, 25.042960341261008] | |
| Bootstrap | wd_95ci | OK | [64.63714329590792, 65.1328029772675] | |
| Bootstrap | theta_95ci | OK | [21.035346173643454, 22.715933599040003] | |
| fx sensitivity | WD range | OK | 58.3 to 71.3 mm | |
| Pose sweep N=3 | rms | OK | 1.02 px | |
| Pose sweep N=5 | rms | OK | 1.03 px | |
| Pose sweep N=8 | rms | OK | 1.06 px | |
| Pose sweep N=10 | rms | OK | 1.07 px | |
| Synth CMO | coupling_norm | OK | 0.8108234887944079 | expected 0.81 |
| BIC | cmo_telecentric_shear | OK | bic=-36128 | |
| BIC | cmo_telecentric | OK | bic=-33201 | |
| BIC | pinhole_parallel_plate | OK | bic=5079 | |
| BIC | central_pinhole | OK | bic=6500 | |
| BIC | central_brown_conrady | OK | bic=6549 | |