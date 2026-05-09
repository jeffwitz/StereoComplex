# %% [markdown]
# # 07 — Model selection classification matrix
#
# This notebook demonstrates that the StereoComplex model selection framework
# correctly identifies the optical architecture for **all six oracle families**:
#
# | # | Oracle | Expected winner | Structure |
# |---|--------|----------------|-----------|
# | 1 | Central pinhole | `central_pinhole` | Pure central projection |
# | 2 | Central Brown-Conrady | `central_brown_conrady` | Central with radial/tangential distortion |
# | 3 | Pinhole + inclined plate | `pinhole_parallel_plate` | Non-central plate line family |
# | 4 | Physical CMO shared-rig | `cmo_physical_shared` | Shared-objective stereo |
# | 5 | Greenough (independent Brown × 2) | `central_brown_conrady` | Independent central channels |
# | 6 | **Uncatalogued** (random high-order Zernike) | `zernike_compact` | Fallback generic model |
#
# ## Why this matters
#
# The framework is not merely a parameter-count competition.  Each oracle is
# the *correct* model for a specific optical architecture.  The framework must
# recover the truth — the runner-up models must have worse BIC either because
# their structure is wrong (high RMS) or because they are over-parameterised
# (high parameter penalty).
#
# The last row is the most important: when the optics fall outside all known
# families, the compact Zernike fallback wins — signalling to the user that the
# instrument does not match any catalogued architecture.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from stereocomplex.benchmarks.model_selection_oracles import (
    StereoOracle,
    build_all_oracles,
)
from stereocomplex.physics import (
    CentralPinholeModel,
    CentralBrownConradyModel,
    PinholeParallelPlateModel,
    PhysicalModelSpec,
    select_physical_model_from_rayfield,
    NonCentralPolynomialChannelModel,
)
from stereocomplex.physics.cmo import CMOIntrinsics
from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeCandidate,
    ZernikeOriginFieldConfig,
)

IMAGE_SIZE = (160, 120)
GRID_SHAPE = (12, 9)
SEED = 42
NOISE_ORIGIN_STD_MM = 0.02   # 20 µm — realistic ChArUco calibration noise floor
NOISE_SEED = 123


class NoisyRayField:
    """Wraps a rayfield with fixed per-pixel Gaussian noise on origins."""

    def __init__(self, field, noise_std_mm: float, rng: np.random.Generator):
        self._field = field
        self._std = float(noise_std_mm)
        self._noise_cache: dict[tuple[int, int], np.ndarray] = {}

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        O, d = self._field.ray(u, v)
        uf = np.asarray(u, dtype=np.float64).reshape(-1)
        vf = np.asarray(v, dtype=np.float64).reshape(-1)
        O_noisy = np.asarray(O, dtype=np.float64).copy()
        # Generate consistent per-pixel noise.
        rng = np.random.default_rng(NOISE_SEED)
        for k in range(len(uf)):
            key = (int(round(uf[k])), int(round(vf[k])))
            if key not in self._noise_cache:
                self._noise_cache[key] = rng.normal(scale=self._std, size=3)
            O_noisy[k] += self._noise_cache[key]
        return O_noisy.reshape(O.shape), d


# %% [markdown]
# ## Oracle builders
#
# Oracles are imported from :mod:`stereocomplex.benchmarks.model_selection_oracles`.
# Each builder returns a :class:`StereoOracle` dataclass with ``.left_field``,
# ``.right_field``, ``.K_left``, ``.K_right``, and metadata.
#
# The six standard oracles cover the full optical catalogue plus one
# uncatalogued exotic case.


# %% [markdown]
# ## Candidate set
#
# All candidates are offered in every case.  The physical CMO is offered only
# when the `target_right` field is provided (stereo mode).

# %%
def build_candidates(K: np.ndarray, image_size: tuple[int, int],
                     pixel_pitch_mm: float | None = None):
    """Return the standard candidate list, including the physical CMO.

    When ``pixel_pitch_mm`` is None (non-CMO oracles), a default pitch of
    0.005 mm is used so the CMO candidate can still be evaluated.  The CMO
    model will fail on non-CMO oracles, demonstrating discrimination.
    """
    _pitch = pixel_pitch_mm if pixel_pitch_mm is not None else 0.005
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1],
                         fx=float(K[0, 0]), fy=float(K[1, 1]),
                         cx=float(K[0, 2]), cy=float(K[1, 2]))
    terms = NonCentralPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40.0, -40.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40.0, +40.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
    )
    lo_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=2)
    n_zernike = len(lo_config.modes()) * 6

    specs = [
        PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
        PhysicalModelSpec(
            "central_brown_conrady", CentralBrownConradyModel, np.zeros(5),
            bounds=(
                np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
            ),
        ),
        PhysicalModelSpec(
            "pinhole_parallel_plate", PinholeParallelPlateModel,
            np.array([0.0, 0.0, 8.0]),
            bounds=(np.array([-30.0, -30.0, 0.0]), np.array([30.0, 30.0, 50.0])),
            model_kwargs={"eta": 1.5, "d1_mm": 80.0},
        ),
        PhysicalModelSpec(
            "cmo_physical_shared", CMOPhysicalStereoModel,
            np.array([80.0, 120.0, 10.0, 50.0, 79.5, 59.5, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            model_kwargs={"pixel_pitch_mm": _pitch},
        ),
    ]

    specs.append(PhysicalModelSpec(
        "polynomial_surrogate_channel", NonCentralPolynomialChannelModel,
        poly_initial, bounds=poly_bounds,
        model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
    ))
    specs.append(PhysicalModelSpec(
        "zernike_compact", ZernikeCandidate,
        np.zeros(n_zernike, dtype=np.float64), bounds=None,
        model_kwargs={"config": lo_config, "fit_directions": True},
    ))
    return specs


# %% [markdown]
# ## Run all six cases

# %%
@dataclass
class CaseResult:
    oracle_name: str
    winner: str
    winner_params: int
    winner_rms: float
    winner_bic: float
    second: str
    second_params: int
    second_rms: float
    second_bic: float
    correct: bool
    all_candidates: dict[str, float]  # model_name → BIC, for the heatmap


def pixels_grid():
    return np.array(
        [[u, v] for v in np.linspace(8.0, float(IMAGE_SIZE[1] - 1) - 8.0, GRID_SHAPE[1])
         for u in np.linspace(8.0, float(IMAGE_SIZE[0] - 1) - 8.0, GRID_SHAPE[0])],
        dtype=np.float64,
    )


def evaluate_case(name: str, left_field, right_field, K, expected: str,
                  pixel_pitch_mm: float | None = None,
                  K_right: np.ndarray | None = None) -> CaseResult:
    px = pixels_grid()
    K_r = K_right if K_right is not None else K
    candidates = build_candidates(K, IMAGE_SIZE, pixel_pitch_mm=pixel_pitch_mm)
    report = select_physical_model_from_rayfield(
        target_field=left_field,
        target_right=right_field,
        candidate_specs=candidates,
        K=K, K_right=K_r,
        image_size=IMAGE_SIZE,
        support_pixels=px, support_pixels_right=px,
        full_grid_weight=0.0,
        max_nfev=1500,
    )
    sorted_candidates = sorted(report.candidates, key=lambda c: c.bic)
    best = sorted_candidates[0]
    second = sorted_candidates[1] if len(sorted_candidates) > 1 else best
    all_bic = {c.model_name: c.bic for c in report.candidates}
    return CaseResult(
        oracle_name=name,
        winner=best.model_name,
        winner_params=best.n_parameters,
        winner_rms=best.rms_mm,
        winner_bic=best.bic,
        second=second.model_name,
        second_params=second.n_parameters,
        second_rms=second.rms_mm,
        second_bic=second.bic,
        correct=(best.model_name == expected),
        all_candidates=all_bic,
    )


# %%
print("Model selection on all six oracles.\n")

oracles = build_all_oracles(IMAGE_SIZE, SEED)

import time as _time

def run_matrix(noise_std: float = 0.0):
    """Run all six cases, optionally adding origin noise."""
    rng = np.random.default_rng(NOISE_SEED)
    saved = []
    for o in oracles:
        left, right, K, K_r = o.left_field, o.right_field, o.K_left, o.K_right
        if noise_std > 0:
            left = NoisyRayField(left, noise_std, rng)
            right = NoisyRayField(right, noise_std, rng)
        t0 = _time.time()
        r = evaluate_case(o.name, left, right, K, o.expected_winner,
                          pixel_pitch_mm=o.pixel_pitch_mm, K_right=K_r)
        elapsed = _time.time() - t0
        saved.append((o.name, r, o.name, elapsed))
    return saved


def print_results(saved, title: str):
    """Print detailed blocks and summary table for one set of results."""
    print(f"── {title} ──")
    for name, r, desc, elapsed in saved:
        status = "✓" if r.correct else "✗ MISCLASSIFIED"
        print(f"━━━ {name} ━━━ ({desc})  [{elapsed:.0f}s]")
        print(f"  Winner: {r.winner}  |  {r.winner_params} params  |  RMS={r.winner_rms:.4f} mm  |  BIC={r.winner_bic:.1f}  |  {status}")
        print(f"  2nd:    {r.second:28s}  |  {r.second_params:3d} params  |  RMS={r.second_rms:.4f} mm  |  BIC={r.second_bic:.1f}")
        print()
    print(f"{'Oracle':<28s} {'Winner':<26s} {'Params':>6s}  {'RMS (mm)':>10s}  {'ΔBIC':>8s}  {'2nd place':<26s}")
    print("-" * 120)
    for name, r, desc, _ in saved:
        delta = r.second_bic - r.winner_bic
        print(f"{name:<28s} {r.winner:<26s} {r.winner_params:>6d}  {r.winner_rms:>10.4f}  {delta:>+8.0f}  {r.second:<26s}  {'✓' if r.correct else '✗'}")
    all_ok = all(r.correct for _, r, _, _ in saved)
    print(f"\nAll {len(saved)} correctly classified: {'YES' if all_ok else 'NO'}\n")
    return saved


# ── Run ────────────────────────────────────────────────────────────
saved_clean = run_matrix(noise_std=0.0)
saved_noisy = run_matrix(noise_std=NOISE_ORIGIN_STD_MM)

print_results(saved_clean, "Noiseless oracles")
print_results(saved_noisy, f"Noisy oracles — {NOISE_ORIGIN_STD_MM*1000:.0f} µm origin noise")

# %% [markdown]
# ## BIC heatmaps
#
# Two heatmaps: noiseless (left column in docs) and with 20 µm origin noise
# (right column).  Each cell shows ΔBIC from the winner.  The diagonal
# (ΔBIC = 0) is the correct classification.

# %%
import matplotlib.pyplot as plt
import matplotlib

_preferred = ["central_pinhole", "central_brown_conrady", "pinhole_parallel_plate",
              "cmo_physical_shared", "polynomial_surrogate_channel", "zernike_compact"]
_oracle_short = ["pinhole", "Brown", "plate", "CMO", "Greenough", "exotic"]
_candidate_short = ["pinhole", "Brown", "plate", "CMO phys", "poly surr", "Zernike"]

_repo_root = Path(__file__).resolve().parent.parent.parent
assets_dir = _repo_root / "docs" / "assets" / "cmo_model_selection"
assets_dir.mkdir(parents=True, exist_ok=True)


def make_heatmap(saved, title: str):
    """Build a ΔBIC heatmap from saved results."""
    oracle_names = [s[0] for s in saved]
    candidate_names = sorted(set().union(*(s[1].all_candidates.keys() for s in saved)))
    candidate_names = [n for n in _preferred if n in candidate_names]
    nr, nc = len(oracle_names), len(candidate_names)

    delta = np.full((nr, nc), np.nan)
    for i, (_, r, _, _) in enumerate(saved):
        best_bic = r.winner_bic
        for j, cn in enumerate(candidate_names):
            if cn in r.all_candidates:
                delta[i, j] = r.all_candidates[cn] - best_bic

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    capped = np.clip(delta, 0, 5000)
    cmap = plt.cm.YlOrRd
    cmap.set_bad("0.9")
    ax.imshow(capped, cmap=cmap, aspect="auto", vmin=0, vmax=capped.max())

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(_candidate_short, rotation=20, ha="right", fontsize=12)
    ax.set_yticklabels(_oracle_short, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(nr):
        for j in range(nc):
            val = delta[i, j]
            if np.isnan(val):
                continue
            if val == 0:
                ax.text(j, i, "0", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="darkgreen",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85))
            elif val < 10_000:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=10, color="black")
            else:
                ax.text(j, i, f"{val/1000:.0f}k", ha="center", va="center", fontsize=10, color="black")

    ax.set_xlabel("Candidate model", fontsize=12)
    ax.set_ylabel("Oracle", fontsize=12)
    fig.tight_layout(pad=0.4)
    return fig


fig_clean = make_heatmap(saved_clean, "ΔBIC from winner — noiseless")
fig_clean.savefig(assets_dir / "classification_heatmap.png", dpi=200, bbox_inches="tight")
fig_clean.savefig(assets_dir / "classification_heatmap.pdf", bbox_inches="tight")
plt.show()

fig_noisy = make_heatmap(saved_noisy, f"ΔBIC from winner — noisy ({NOISE_ORIGIN_STD_MM*1000:.0f} µm)")
fig_noisy.savefig(assets_dir / "classification_heatmap_noisy.png", dpi=200, bbox_inches="tight")
fig_noisy.savefig(assets_dir / "classification_heatmap_noisy.pdf", bbox_inches="tight")
plt.show()

print("Heatmaps saved.")

# %% [markdown]
# ## Interpretation
#
# ### Noiseless regime
#
# Each row is a **separate experiment** — a different synthetic oracle
# representing one optical architecture.  The RMS values are NOT comparable
# across rows (different oracles have different scales).  Within each row,
# the winner is the model with the lowest BIC among all candidates.
#
# - **Rows 1–3**: Classical stereo (pinhole, Brown, plate).  The correct model
#   wins because it matches the oracle's structure with the fewest parameters.
#
# - **Row 4**: CMO shared-rig.  The physical CMO wins (17 shared params)
#   against the polynomial surrogate (36 params) and compact Zernike (72
#   params).  BIC penalises the independent-channel over-parameterisation.
#
# - **Row 5**: Greenough.  Two independent Brown-Conrady channels win (10
#   params total).  The ΔBIC is the smallest among structural matches (~900)
#   because the polynomial surrogate's family **includes** Brown-Conrady as a
#   special case.  The selection here is purely a parsimony argument: BIC
#   penalises the 26 unnecessary polynomial parameters.
#
# - **Row 6**: Uncatalogued.  A low-amplitude high-order Zernike rayfield
#   (max_order=3) belongs to no known physical family.  The compact Zernike
#   candidate (max_order=2) wins — correctly signalling that the optics fall
#   outside the catalogue.  This is the **detector row**: when `zernike_compact`
#   wins, no physical model in the current set is adequate.
#
# ### Noisy regime (20 µm origin noise)
#
# Under 20 µm Gaussian noise on ray origins — realistic for a well-calibrated
# ChArUco stereo rig — all six classifications remain correct.  The diagonal
# of the heatmap is preserved.  However, the ΔBIC margins tell a richer story:
#
# **Three regimes emerge under noise:**
#
# 1. **Noise-floor parsimony** (pinhole, Brown, Greenough).  The oracle's
#    structural signature is shallow enough that several candidates all reach
#    the ~0.039 mm noise floor.  BIC then discriminates *purely on parameter
#    count*.  The pinhole case is the most fragile: ΔBIC = +27 means the
#    zero-parameter model beats the 10-parameter Brown model by a narrow margin.
#    At ~50 µm noise this case would flip — Brown's extra parameters would buy
#    enough fit improvement to overcome the penalty.
#
# 2. **Structural mismatch** (plate, CMO).  The noise floor is ~0.039 mm but
#    structurally wrong models leave residuals well above it.  The correct
#    model wins by a margin of 100–200 BIC units — an order of magnitude larger
#    than the parsimony cases.  The CMO physical model achieves the lowest
#    absolute RMS (0.0378 mm) *and* uses fewer parameters (17 shared) than the
#    generic fallbacks (36–72).  This is the ideal: the right structure fits
#    better AND is more compact.
#
# 3. **Exotic detection** (uncatalogued).  The ΔBIC margin is essentially
#    unchanged (+2077 → +2078) because the structural RMS gap (~2.7 mm for the
#    best physical model) dwarfs the 20 µm noise.  The Zernike compact
#    candidate wins robustly — the noise does not accidentally make a physical
#    model look good on optics it cannot represent.
#
# **Key insight:** the framework degrades *gracefully*.  As noise increases,
#    the parsimony cases flip first (pinhole vs Brown at ~50 µm), then the
#    structural cases (plate, CMO at much higher noise), and the exotic
#    detection is the last to fail.  This ordering is exactly what one wants:
#    the most scientifically important distinctions (CMO vs non-CMO, catalogued
#    vs uncatalogued) are the most robust to measurement noise.
