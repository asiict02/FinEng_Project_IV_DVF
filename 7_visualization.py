"""
7_visualization.py — Visualization of Results
  1. IV Surface (3D)
  2. IV Smile by maturity bucket
  3. DVF model fit vs Market IV
  4. OOS Loss Heatmap (2x2 per model)
  5. Pricing residuals by moneyness
"""

import os, sys, importlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_dvf  = importlib.import_module("3_dvf_models")

PROC_DIR = os.path.join(_HERE, "DataSet", "data", "processed")
FIG_DIR  = os.path.join(_HERE, "DataSet", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df_iv      = pd.read_csv(os.path.join(PROC_DIR, "options_with_iv.csv"), parse_dates=["ObsDate","ExDt"])
df_test    = pd.read_csv(os.path.join(PROC_DIR, "options_test.csv"),    parse_dates=["ObsDate","ExDt"])
df_oos     = pd.read_csv(os.path.join(PROC_DIR, "oos_all_losses.csv"))
df_params  = pd.read_csv(os.path.join(PROC_DIR, "fitted_params.csv"))

# Build params dict
params_dict = {}
for _, p in df_params.iterrows():
    m, l = p["model_id"], p["loss_id"]
    params_dict.setdefault(m, {})[l] = np.array([p[n] for n in _dvf.MODEL_SPECS[m]["param_names"]])

COLORS  = {"M0":"#888888","M1":"#1A73E8","M2":"#E84040","M3":"#2CA02C","M4":"#FF7F0E"}
BUCKETS = {"1M":(0.04,0.12),"3M":(0.12,0.30),"6M":(0.30,0.60),"12M":(0.60,1.10)}
GRID    = np.linspace(0.70, 1.30, 200)

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {name}")

# ── Plot 1: IV Surface ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10,7))
ax  = fig.add_subplot(111, projection="3d")
sc  = ax.scatter(df_iv["Moneyness"], df_iv["T"], df_iv["IV"],
                 c=df_iv["IV"], cmap="RdYlGn_r", alpha=0.4, s=3)
ax.set(xlabel="Moneyness (K/S)", ylabel="Maturity (yrs)", zlabel="IV")
ax.set_title("Market IV Surface — SPX")
fig.colorbar(sc, ax=ax, shrink=0.5)
save("01_iv_surface.png")

# ── Plot 2: IV Smile by maturity bucket ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16,4), sharey=True)
for ax, (lbl,(lo,hi)) in zip(axes, BUCKETS.items()):
    sub = df_iv[(df_iv["T"]>=lo)&(df_iv["T"]<hi)]
    for ot, c, mk in [("call","#1A73E8","o"),("put","#E84040","s")]:
        g = sub[sub["OptionType"]==ot]
        ax.scatter(g["Moneyness"], g["IV"], c=c, marker=mk, s=8, alpha=0.4, label=ot)
    ax.axvline(1.0, color="k", ls="--", alpha=0.3)
    ax.set(title=lbl, xlabel="Moneyness (K/S)")
    ax.legend(fontsize=7)
axes[0].set_ylabel("IV")
fig.suptitle("IV Smile by Maturity — SPX", y=1.02)
plt.tight_layout()
save("02_iv_smile.png")

for loss in ["L2", "L5"]:
    # ── Plot 3: DVF fit vs Market IV ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16,4), sharey=True)
    for ax, (lbl,(lo,hi)) in zip(axes, BUCKETS.items()):
        sub = df_test[(df_test["T"]>=lo)&(df_test["T"]<hi)]
        if sub.empty: continue
        ax.scatter(sub["Moneyness"], sub["IV"], c="#aaaaaa", s=6, alpha=0.3, label="Market")
        T_mid = float(sub["T"].median())
        for m, col in COLORS.items():
            sig = _dvf.predict_sigma(params_dict[m][loss], GRID, np.full_like(GRID,T_mid), m) * np.ones_like(GRID)
            ax.plot(GRID, sig, color=col, lw=1.8, label=m)
        ax.axvline(1.0, color="k", ls="--", alpha=0.3)
        ax.set(title=lbl, xlabel="Moneyness (K/S)", ylim=(0.05, 0.65))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("IV")
    fig.suptitle(f"DVF Model Fit vs Market IV (est. {loss})", y=1.02)
    plt.tight_layout()
    save(f"03_dvf_fit_{loss}.png")

    # ── Plot 4: Loss Heatmap ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(16,3.5))
    for ax, m in zip(axes, _dvf.MODEL_SPECS):
        col = (df_oos[df_oos["model_id"]==m].set_index("est_loss")[[loss]].reindex(["L2","L5"])).astype(float)
        sns.heatmap(col, ax=ax, annot=True, fmt=".5f", cmap="YlOrRd",
                    linewidths=0.5, cbar=False, xticklabels=[loss], yticklabels=["L2","L5"])
        ax.add_patch(plt.Rectangle((-0.5,["L2","L5"].index(loss)-0.5),1,1,fill=False,edgecolor="k",lw=2.5))
        ax.set_title(f"Model {m}", fontweight="bold")
        ax.set_xlabel(f"Eval: {loss}")
        ax.set_ylabel("Est Loss" if ax==axes[0] else "")
    fig.suptitle(f"OOS Loss Heatmap — Eval {loss}  (■ = matched est/eval)", y=1.04)
    plt.tight_layout()
    save(f"04_loss_heatmap_{loss}.png")

    # ── Plot 5: Pricing residuals ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20,4), sharey=True)
    for ax, (m, col) in zip(axes, COLORS.items()):
        df_pred = _dvf.apply_model_to_df(df_test, params_dict[m][loss], m)
        res = df_pred["ModelSigma"] - df_pred["IV"]
        ax.scatter(df_pred["Moneyness"], res, c=col, s=5, alpha=0.3)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.axvline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{m}  RMSE={float(np.sqrt((res**2).mean())):.4f}")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylim(-0.15, 0.10)
    axes[0].set_ylabel("Model IV − Market IV")
    fig.suptitle(f"IV Residuals by Moneyness (est. {loss})", y=1.02)
    plt.tight_layout()
    save(f"05_residuals_{loss}.png")

print(f"\nAll figures saved to: {FIG_DIR}")
