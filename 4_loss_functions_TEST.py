"""
4_loss_functions.py
===================
Step 4: Loss Functions — Christoffersen & Jacobs (2004)

L2: IV-MSE       — mean( (IV_model - IV_market)^2 )
L5: Vega-IVMSE   — mean( Vega^2 * (IV_model - IV_market)^2 )

Imported by: 5_estimation.py
"""

import numpy as np

# Iterable by estimation.py
LOSS_FUNCTIONS = {
    "L2": "IV-MSE      — mean( (IV_model - IV_market)^2 )",
    "L5": "Vega-IVMSE  — mean( Vega^2 * (IV_model - IV_market)^2 )",
}


def compute_loss(loss_id: str, model_ivs: np.ndarray, market_ivs: np.ndarray,
                 vegas: np.ndarray = None) -> float:
    """Dispatcher: returns the scalar loss for loss_id in {'L2', 'L5'}."""
    err2 = (model_ivs - market_ivs) ** 2
    if loss_id == "L2":
        return float(np.mean(err2))
    elif loss_id == "L5":
        if vegas is None:
            raise ValueError("vegas array required for L5.")
        return float(np.mean(vegas**2 * err2))
    raise ValueError(f"Unknown loss_id '{loss_id}'. Use 'L2' or 'L5'.")


def compute_all_losses(model_ivs: np.ndarray, market_ivs: np.ndarray,
                       vegas: np.ndarray) -> dict:
    """Returns both L2 and L5 in one call."""
    err2 = (model_ivs - market_ivs) ** 2
    return {
        "L2": float(np.mean(err2)),
        "L5": float(np.mean(vegas**2 * err2)),
    }

if __name__ == "__main__":
    import sys
    print("\n=== 4_loss_functions.py — Diagnostic Test ===\n")
    failures = []

    # ── TEST 1: Import check ───────────────────────────────────────────────────
    # If we're here, numpy loaded and the module is syntactically valid.
    print("[1/5] Import check .............. OK  (numpy loaded, module is valid)")

    # ── TEST 2: L2 correctness — known analytic answer ────────────────────────
    # If model_ivs = market_ivs + 0.01 for all options, then:
    #   L2 = mean((0.01)^2) = 0.0001 exactly.
    # Any other answer means the formula is wrong.
    print("\n[2/5] L2 correctness — known analytic answer:")
    N          = 100
    market_ivs = np.full(N, 0.20)       # flat market IV of 20%
    model_ivs  = np.full(N, 0.21)       # model overshoots by exactly 1%
    expected   = 0.01**2                 # = 0.0001
    result     = compute_loss("L2", model_ivs, market_ivs)
    error      = abs(result - expected)
    status     = "OK" if error < 1e-10 else "FAIL"
    print(f"      Expected : {expected:.10f}")
    print(f"      Got      : {result:.10f}")
    print(f"      Error    : {error:.2e}  [{status}]")
    if status == "FAIL":
        failures.append("L2 analytic answer wrong")

    # ── TEST 3: L5 correctness — known analytic answer ────────────────────────
    # Same setup, but now vega = 2.0 for all options.
    #   L5 = mean(2^2 * (0.01)^2) = mean(4 * 0.0001) = 0.0004 exactly.
    print("\n[3/5] L5 correctness — known analytic answer:")
    vegas    = np.full(N, 2.0)
    expected = (2.0**2) * (0.01**2)     # = 0.0004
    result   = compute_loss("L5", model_ivs, market_ivs, vegas)
    error    = abs(result - expected)
    status   = "OK" if error < 1e-10 else "FAIL"
    print(f"      Expected : {expected:.10f}")
    print(f"      Got      : {result:.10f}")
    print(f"      Error    : {error:.2e}  [{status}]")
    if status == "FAIL":
        failures.append("L5 analytic answer wrong")

    # ── TEST 4: Boundary and consistency checks ────────────────────────────────
    print("\n[4/5] Boundary and consistency checks:")

    np.random.seed(42)
    market_ivs_r = np.random.uniform(0.10, 0.40, 500)
    model_ivs_r  = market_ivs_r + np.random.normal(0, 0.02, 500)
    vegas_r      = np.random.uniform(0.5, 50.0, 500)

    l2 = compute_loss("L2", model_ivs_r, market_ivs_r)
    l5 = compute_loss("L5", model_ivs_r, market_ivs_r, vegas_r)

    checks = {
        "L2 > 0 (errors exist)":             l2 > 0,
        "L5 > 0 (errors exist)":             l5 > 0,
        "L2 is finite":                       np.isfinite(l2),
        "L5 is finite":                       np.isfinite(l5),
        "L5 > L2 (vega scaling inflates it)": l5 > l2,
        "Perfect fit → L2 = 0":              compute_loss("L2", market_ivs_r, market_ivs_r) == 0.0,
        "Perfect fit → L5 = 0":              compute_loss("L5", market_ivs_r, market_ivs_r, vegas_r) == 0.0,
    }

    for desc, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"      {desc:<40} [{status}]")
        if not passed:
            failures.append(desc)

    # ── TEST 5: compute_all_losses consistency ─────────────────────────────────
    # compute_all_losses must return exactly the same values as calling
    # compute_loss('L2') and compute_loss('L5') individually.
    print("\n[5/5] compute_all_losses consistency:")
    all_losses = compute_all_losses(model_ivs_r, market_ivs_r, vegas_r)

    match_l2 = abs(all_losses["L2"] - l2) < 1e-12
    match_l5 = abs(all_losses["L5"] - l5) < 1e-12
    has_keys  = set(all_losses.keys()) == {"L2", "L5"}

    for desc, passed in [
        ("Keys are exactly {'L2', 'L5'}", has_keys),
        ("L2 matches compute_loss('L2')", match_l2),
        ("L5 matches compute_loss('L5')", match_l5),
    ]:
        status = "OK" if passed else "FAIL"
        print(f"      {desc:<40} [{status}]")
        if not passed:
            failures.append(desc)

    # ── Error handling checks ──────────────────────────────────────────────────
    print("\n      Error handling:")
    try:
        compute_loss("L5", model_ivs_r, market_ivs_r, vegas=None)
        print("      Missing vegas for L5 .................... [FAIL — should have raised]")
        failures.append("L5 missing vegas not caught")
    except ValueError:
        print("      Missing vegas for L5 raises ValueError . [OK]")

    try:
        compute_loss("L9", model_ivs_r, market_ivs_r)
        print("      Unknown loss_id raises ValueError ....... [FAIL — should have raised]")
        failures.append("Unknown loss_id not caught")
    except ValueError:
        print("      Unknown loss_id raises ValueError ....... [OK]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    if not failures:
        print("=== Diagnostic complete. All [OK] = ready for estimation.py ===\n")
    else:
        print(f"=== {len(failures)} check(s) FAILED ===")
        for f in failures:
            print(f"    ✗ {f}")
        sys.exit(1)