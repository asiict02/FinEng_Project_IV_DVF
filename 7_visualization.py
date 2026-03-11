"""
visualization.py
================
Step 7: Visualisation

Produces all plots needed for the report:
  1. IV Surface          — 3D scatter of market-implied vols across (K/S, T)
  2. IV Smile Plots      — IV vs moneyness for fixed maturity buckets
  3. DVF Fit Overlay     — predicted vs actual IV smile for each model M0–M4
  4. Loss Heatmap        — 5×5 OOS loss matrix as a colour map
  5. Residual Plot       — pricing errors across moneyness and maturity

Inputs:
    data/processed/options_with_iv.csv     (IV data)
    data/processed/options_test.csv        (test set)
    data/processed/fitted_params.csv       (fitted parameters)
    data/processed/oos_all_losses.csv      (OOS evaluation results)

Outputs:
    outputs/figures/iv_surface.png
    outputs/figures/iv_smile.png
    outputs/figures/dvf_fit_overlay.png
    outputs/figures/loss_heatmap.png
    outputs/figures/residuals.png

Usage:
    python visualization.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — needed for 3D projection

# seaborn makes heatmaps easy — import with fallback if not installed
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠ seaborn not installed. Heatmap will use matplotlib fallback.")

from dvf_models     import apply_model_to_df, MODEL_SPECS
from loss_functions import LOSS_FUNCTIONS
from evaluation     import load_fitted_params

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

IV_PATH      = "data/processed/options_with_iv.csv"
TEST_PATH    = "data/processed/options_test.csv"
PARAMS_PATH  = "data/processed/fitted_params.csv"
OOS_PATH     = "data/processed/oos_all_losses.csv"
OUTPUT_DIR   = "outputs/figures"

# Colour palette for models M0–M4
MODEL_COLORS = {
    "M0": "#999999",   # grey   — flat benchmark
    "M1": "#1A73E8",   # blue   — linear
    "M2": "#E84040",   # red    — quadratic
    "M3": "#2CA02C",   # green  — quad + maturity
    "M4": "#FF7F0E",   # orange — full model
}

# Which estimation loss to use for the DVF overlay plots (Section 3)
# L5 (vega-weighted) is the recommended one from Christoffersen & Jacobs
OVERLAY_LOSS = "L5"

# Maturity buckets for smile plots (in years)
# e.g. "1M" = options with T between 0.04 and 0.12 years (roughly 1 month)
MATURITY_BUCKETS = {
    "1M":  (0.04, 0.12),
    "3M":  (0.12, 0.30),
    "6M":  (0.30, 0.60),
    "12M": (0.60, 1.10),
}


# =============================================================================
# SECTION 2: SETUP
# =============================================================================

def setup():
    """Creates output directory and sets matplotlib style."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use a clean style — readable for academic papers
    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  9,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "lines.linewidth":  1.8,
    })


# =============================================================================
# SECTION 3: PLOT 1 — IV SURFACE (3D)
# =============================================================================

def plot_iv_surface(df: pd.DataFrame) -> None:
    """
    Plots the market-implied volatility surface as a 3D scatter plot.

    X-axis: Moneyness = K/S (so 1.0 = at-the-money)
    Y-axis: Time to maturity T (in years)
    Z-axis: Implied volatility IV (as decimal, e.g. 0.20 = 20%)

    What to look for in the report:
      - The smile/skew: IV is typically higher for low moneyness (OTM puts)
        than for high moneyness (OTM calls) — this is the "volatility skew"
        observed in equity markets since the 1987 crash.
      - The term structure: IV often increases with maturity (but can invert
        during stress periods like COVID).
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Colour each point by IV level — higher IV = warmer colour
    scatter = ax.scatter(
        df["Moneyness"],    # x
        df["T"],            # y
        df["IV"],           # z
        c     = df["IV"],   # colour by IV value
        cmap  = "RdYlGn_r", # red = high vol, green = low vol
        alpha = 0.5,
        s     = 5,          # small dot size
    )

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Time to Maturity (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Market Implied Volatility Surface — SPX")
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="IV")

    path = os.path.join(OUTPUT_DIR, "iv_surface.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved IV surface → {path}")


# =============================================================================
# SECTION 4: PLOT 2 — IV SMILE (2D, by maturity bucket)
# =============================================================================

def plot_iv_smile(df: pd.DataFrame) -> None:
    """
    Plots IV vs moneyness (K/S) for each maturity bucket in one figure.

    Each subplot = one maturity bucket (1M, 3M, 6M, 12M).
    Each dot = one observed option.

    What to look for:
      - Downward sloping smile (skew): typical for SPX, where OTM puts
        carry higher IV than OTM calls (crash insurance premium).
      - How the smile shape changes with maturity — longer-dated options
        tend to have a flatter smile.
    """
    n_buckets = len(MATURITY_BUCKETS)
    fig, axes = plt.subplots(1, n_buckets, figsize=(4 * n_buckets, 4), sharey=True)

    for ax, (label, (T_lo, T_hi)) in zip(axes, MATURITY_BUCKETS.items()):
        # Filter to this maturity bucket
        mask   = (df["T"] >= T_lo) & (df["T"] < T_hi)
        subset = df[mask]

        if subset.empty:
            ax.set_title(f"{label}\n(no data)")
            continue

        # Scatter calls and puts in different colours
        for opt_type, color, marker in [("call", "#1A73E8", "o"), ("put", "#E84040", "s")]:
            sub = subset[subset["OptionType"] == opt_type]
            ax.scatter(sub["Moneyness"], sub["IV"],
                       c=color, marker=marker, s=10, alpha=0.5, label=opt_type)

        ax.set_title(f"Maturity: {label}")
        ax.set_xlabel("Moneyness (K/S)")
        ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.legend()

    axes[0].set_ylabel("Implied Volatility")
    fig.suptitle("Implied Volatility Smile by Maturity Bucket — SPX", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "iv_smile.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved IV smile → {path}")


# =============================================================================
# SECTION 5: PLOT 3 — DVF FIT OVERLAY
# =============================================================================

def plot_dvf_overlay(df_test: pd.DataFrame, params_dict: dict) -> None:
    """
    For each maturity bucket, plots:
      - Grey dots  = actual market IVs
      - Coloured lines = DVF model predictions for M0–M4

    Uses the parameters estimated with OVERLAY_LOSS (default: L5).
    One subplot per maturity bucket.

    What to look for:
      - M0 (flat) will be a horizontal line — clearly misses the smile shape.
      - M1/M2 start capturing the slope/curvature of the smile.
      - M3/M4 additionally capture the term structure.
      - Discussion point: which model fits best, and does it vary by maturity?
    """
    n_buckets = len(MATURITY_BUCKETS)
    fig, axes = plt.subplots(1, n_buckets, figsize=(4 * n_buckets, 4), sharey=True)

    for ax, (label, (T_lo, T_hi)) in zip(axes, MATURITY_BUCKETS.items()):
        mask   = (df_test["T"] >= T_lo) & (df_test["T"] < T_hi)
        subset = df_test[mask].sort_values("Moneyness")

        if subset.empty:
            ax.set_title(f"{label}\n(no data)")
            continue

        # Plot raw market IV as grey background dots
        ax.scatter(subset["Moneyness"], subset["IV"],
                   color="grey", s=8, alpha=0.4, label="Market IV", zorder=1)

        # Overlay each DVF model's predicted sigma along the moneyness axis
        moneyness_grid = np.linspace(subset["Moneyness"].min(),
                                     subset["Moneyness"].max(), 100)
        T_mid = (T_lo + T_hi) / 2   # representative maturity for this bucket

        for model_id, color in MODEL_COLORS.items():
            # Get the params estimated with OVERLAY_LOSS for this model
            params = params_dict[model_id][OVERLAY_LOSS]

            # Convert moneyness back to strike: K = moneyness * S
            # Use the mean S0 in this bucket as representative
            S_mean = subset["S0"].mean()
            K_grid = moneyness_grid * S_mean

            # Predict sigma at each point on the grid
            sigma_grid = np.array([
                predict_sigma_wrapper(params, K, T_mid, model_id)
                for K in K_grid
            ])

            ax.plot(moneyness_grid, sigma_grid,
                    color=color, linewidth=1.8,
                    label=model_id, zorder=2)

        ax.set_title(f"Maturity: {label}")
        ax.set_xlabel("Moneyness (K/S)")
        ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Implied Volatility")
    fig.suptitle(f"DVF Model Fit vs Market IV (est. with {OVERLAY_LOSS})", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "dvf_fit_overlay.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved DVF overlay → {path}")


def predict_sigma_wrapper(params, K, T, model_id):
    """Small helper to avoid importing predict_sigma at the top (circular risk)."""
    from dvf_models import predict_sigma
    return predict_sigma(params, K, T, model_id)


# =============================================================================
# SECTION 6: PLOT 4 — OOS LOSS HEATMAP (the Christoffersen & Jacobs table)
# =============================================================================

def plot_loss_heatmap(df_results: pd.DataFrame) -> None:
    """
    Plots the 5×5 OOS loss matrix as a colour-coded heatmap.

    Rows    = estimation loss (used when fitting the model)
    Columns = evaluation loss (used when judging OOS performance)
    Colour  = OOS loss value (darker = lower = better)

    One heatmap per DVF model (M0–M4) → 5 subplots.

    This is the visual version of Table 3 in Christoffersen & Jacobs (2004).
    The key observation: if all off-diagonal cells were similar to the diagonal,
    loss function choice wouldn't matter. If they differ a lot, it does matter.
    """
    model_ids = list(MODEL_SPECS.keys())
    loss_ids  = list(LOSS_FUNCTIONS.keys())

    fig, axes = plt.subplots(1, len(model_ids), figsize=(4 * len(model_ids), 4))

    for ax, model_id in zip(axes, model_ids):
        # Filter to this model and build the 5×5 matrix
        df_model = df_results[df_results["model_id"] == model_id]
        matrix   = df_model.pivot(
            index   = "est_loss_id",
            columns = None,
            values  = None,
        )
        # Extract the 5×5 sub-matrix of OOS losses
        matrix = (df_model
                  .set_index("est_loss_id")[loss_ids]
                  .reindex(loss_ids))
        matrix_vals = matrix.values.astype(float)

        if HAS_SEABORN:
            # seaborn heatmap: annotate each cell with the numeric value
            sns.heatmap(
                matrix_vals,
                ax          = ax,
                annot       = True,
                fmt         = ".4f",
                cmap        = "YlOrRd",     # yellow = low loss (good), red = high (bad)
                linewidths  = 0.5,
                xticklabels = loss_ids,
                yticklabels = loss_ids,
                cbar        = False,
            )
        else:
            # matplotlib fallback if seaborn not installed
            im = ax.imshow(matrix_vals, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(loss_ids)))
            ax.set_xticklabels(loss_ids)
            ax.set_yticks(range(len(loss_ids)))
            ax.set_yticklabels(loss_ids)
            # Annotate each cell
            for i in range(len(loss_ids)):
                for j in range(len(loss_ids)):
                    ax.text(j, i, f"{matrix_vals[i,j]:.4f}",
                            ha="center", va="center", fontsize=7)

        ax.set_title(f"Model {model_id}")
        ax.set_xlabel("Evaluation Loss")
        ax.set_ylabel("Estimation Loss")

        # Mark diagonal cells with a box (matched estimation/evaluation)
        for i in range(len(loss_ids)):
            ax.add_patch(plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=2
            ))

    fig.suptitle("OOS Loss Matrix — Estimation Loss vs Evaluation Loss\n"
                 "(Diagonal ■ = matched; off-diagonal = cross-evaluation)",
                 fontsize=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "loss_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved loss heatmap → {path}")


# =============================================================================
# SECTION 7: PLOT 5 — RESIDUAL PLOT
# =============================================================================

def plot_residuals(df_test: pd.DataFrame, params_dict: dict) -> None:
    """
    Plots pricing residuals (ModelPrice - MarketPrice) vs moneyness
    for each DVF model, using parameters estimated with L5.

    One subplot per model (M0–M4).

    What to look for:
      - M0 (flat vol) should show systematic patterns: e.g. positive errors
        for OTM puts (model underprices them because it ignores the skew).
      - Better models should show residuals scattered randomly around 0.
      - Any remaining pattern = systematic mis-pricing = room for improvement.
    """
    n_models = len(MODEL_SPECS)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)

    for ax, (model_id, color) in zip(axes, MODEL_COLORS.items()):
        params   = params_dict[model_id][OVERLAY_LOSS]
        df_pred  = apply_model_to_df(df_test, params, model_id)

        # Compute residual: model price minus market price
        # Positive = model over-prices; Negative = model under-prices
        residuals = df_pred["ModelPrice"] - df_pred["MidPrice"]

        # Scatter residuals against moneyness
        ax.scatter(df_pred["Moneyness"], residuals,
                   c=color, s=6, alpha=0.4)

        # Horizontal line at 0 = perfect pricing
        ax.axhline(0, color="black", linewidth=1, linestyle="--")

        # Vertical line at moneyness=1 = ATM
        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")

        # Annotate with RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))
        ax.set_title(f"Model {model_id}\nRMSE = {rmse:.2f}")
        ax.set_xlabel("Moneyness (K/S)")

    axes[0].set_ylabel("Residual (ModelPrice − MarketPrice)")
    fig.suptitle(f"Pricing Residuals by Moneyness (est. with {OVERLAY_LOSS})", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "residuals.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved residuals → {path}")


# =============================================================================
# SECTION 8: MAIN
# =============================================================================

def main():
    print("\n=== Step 7: Visualisation ===\n")
    setup()

    # ── Load all datasets ──────────────────────────────────────────────────────
    print("Loading data...")
    df_iv      = pd.read_csv(IV_PATH,   parse_dates=["ObsDate", "ExDt"])
    df_test    = pd.read_csv(TEST_PATH, parse_dates=["ObsDate", "ExDt"])
    df_results = pd.read_csv(OOS_PATH)
    params_dict = load_fitted_params(PARAMS_PATH)
    print(f"  IV data:   {len(df_iv)} rows")
    print(f"  Test data: {len(df_test)} rows\n")

    # ── Generate all plots ─────────────────────────────────────────────────────
    print("Generating plots...")

    # Plot 1: Raw IV surface
    plot_iv_surface(df_iv)

    # Plot 2: IV smile by maturity bucket
    plot_iv_smile(df_iv)

    # Plot 3: DVF model fit overlay on smile
    plot_dvf_overlay(df_test, params_dict)

    # Plot 4: OOS loss heatmap (the main Christoffersen & Jacobs result)
    plot_loss_heatmap(df_results)

    # Plot 5: Pricing residuals by moneyness
    plot_residuals(df_test, params_dict)

    print(f"\n✓ All plots saved to {OUTPUT_DIR}/")
    print("\n=== Done. Project pipeline complete. ===")
    print("\nRun order summary:")
    print("  1. python data_collection.py")
    print("  2. python implied_vol.py")
    print("  3. (dvf_models.py is imported, not run directly)")
    print("  4. (loss_functions.py is imported, not run directly)")
    print("  5. python estimation.py")
    print("  6. python evaluation.py")
    print("  7. python visualization.py")


if __name__ == "__main__":
    main()
