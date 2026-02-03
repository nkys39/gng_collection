"""Test Lazy Error equivalence with Standard GNG.

Verifies whether the paper's lazy error formulation produces
equivalent results to standard GNG's error handling.
"""

import numpy as np


def standard_gng_error_update(
    error: float, dist_sq: float, beta: float, remaining_steps: int
) -> float:
    """Standard GNG: decay all, then add, then decay for remaining steps.

    Standard GNG per step:
        1. E = E * (1 - beta)  for ALL nodes
        2. E_winner += dist_sq

    So after winning at step s with (lambda - s) remaining steps:
        E = (E * (1-beta) + dist_sq) * (1-beta)^(remaining_steps - 1)

    But actually in demogng order: decay first, then add
        Step s: E = E * (1-beta), then E += dist_sq
        Step s+1 to lambda-1: E = E * (1-beta)^(remaining_steps - 1)

    Final: E = (E * (1-beta) + dist_sq) * (1-beta)^(remaining_steps - 1)
    """
    # Decay existing error
    error = error * (1 - beta)
    # Add new error
    error = error + dist_sq
    # Decay for remaining steps (except current)
    if remaining_steps > 1:
        error = error * ((1 - beta) ** (remaining_steps - 1))
    return error


def lazy_error_update(
    error: float, dist_sq: float, beta: float, lambda_: int, step: int
) -> float:
    """Paper's Lazy Error (Algorithm 4).

    inc_error*(c, s, v, value):
        E_v = beta^(lambda - s) * E_v + value
    """
    decay_factor = beta ** (lambda_ - step)
    return decay_factor * error + dist_sq


def test_single_win():
    """Test error after a single win at different steps."""
    print("=" * 60)
    print("Test: Single win at step s")
    print("=" * 60)

    E_init = 10.0
    dist_sq = 5.0
    lambda_ = 100

    # Standard GNG uses small beta (decay rate)
    beta_std = 0.0005
    # Paper uses large beta (decay factor)
    beta_paper = 1 - beta_std  # = 0.9995

    print(f"E_init = {E_init}, dist_sq = {dist_sq}, lambda = {lambda_}")
    print(f"beta_std = {beta_std}, beta_paper = {beta_paper}")
    print()

    print(f"{'Step s':<10} {'Standard GNG':<20} {'Lazy Error':<20} {'Difference':<15}")
    print("-" * 65)

    for step in [0, 25, 50, 75, 99]:
        remaining = lambda_ - step

        E_std = standard_gng_error_update(E_init, dist_sq, beta_std, remaining)
        E_lazy = lazy_error_update(E_init, dist_sq, beta_paper, lambda_, step)

        diff = E_lazy - E_std
        diff_pct = (diff / E_std) * 100 if E_std != 0 else 0

        print(f"{step:<10} {E_std:<20.6f} {E_lazy:<20.6f} {diff:+.6f} ({diff_pct:+.2f}%)")


def test_multiple_wins():
    """Test error accumulation with multiple wins."""
    print()
    print("=" * 60)
    print("Test: Multiple wins in one cycle")
    print("=" * 60)

    lambda_ = 100
    beta_std = 0.0005
    beta_paper = 1 - beta_std

    # Simulate winning at steps 0, 25, 50, 75
    win_steps = [0, 25, 50, 75]
    dist_sq_values = [5.0, 3.0, 4.0, 2.0]

    print(f"Win at steps: {win_steps}")
    print(f"dist_sq values: {dist_sq_values}")
    print()

    # Standard GNG simulation
    E_std = 10.0
    for i, step in enumerate(win_steps):
        # Decay from previous step to this step
        if i == 0:
            steps_since_last = step + 1  # From cycle start
        else:
            steps_since_last = step - win_steps[i-1]

        # Decay
        E_std = E_std * ((1 - beta_std) ** steps_since_last)
        # Add error
        E_std = E_std + dist_sq_values[i]

    # Final decay to end of cycle
    remaining = lambda_ - win_steps[-1] - 1
    if remaining > 0:
        E_std = E_std * ((1 - beta_std) ** remaining)

    # Lazy Error simulation
    E_lazy = 10.0
    for i, step in enumerate(win_steps):
        decay_factor = beta_paper ** (lambda_ - step)
        E_lazy = decay_factor * E_lazy + dist_sq_values[i]

    print(f"Standard GNG final error: {E_std:.6f}")
    print(f"Lazy Error final error:   {E_lazy:.6f}")
    print(f"Difference: {E_lazy - E_std:+.6f} ({(E_lazy - E_std) / E_std * 100:+.2f}%)")


def test_no_wins():
    """Test error decay for nodes that don't win."""
    print()
    print("=" * 60)
    print("Test: Node that never wins (1 cycle)")
    print("=" * 60)

    E_init = 10.0
    lambda_ = 100
    beta_std = 0.0005
    beta_paper = 1 - beta_std

    # Standard: decay every step
    E_std = E_init * ((1 - beta_std) ** lambda_)

    # Lazy: fix_error applies decay at cycle boundary
    # E = beta^(lambda * 1) * E = beta^lambda * E
    E_lazy = E_init * (beta_paper ** lambda_)

    print(f"E_init = {E_init}, lambda = {lambda_}")
    print(f"Standard GNG: {E_std:.6f}")
    print(f"Lazy Error (fix_error): {E_lazy:.6f}")
    print(f"Difference: {E_lazy - E_std:+.6f} ({(E_lazy - E_std) / E_std * 100:+.2f}%)")


def analyze_formula_difference():
    """Analyze the mathematical difference."""
    print()
    print("=" * 60)
    print("Mathematical Analysis")
    print("=" * 60)

    print("""
Standard GNG (winner at step s):
    E_final = E_prev * (1-β_std)^(λ-s) + dist_sq * (1-β_std)^(λ-s-1)

Lazy Error (winner at step s):
    E_final = β_paper^(λ-s) * E_prev + dist_sq

With β_paper = (1 - β_std):
    - E_prev term: β_paper^(λ-s) = (1-β_std)^(λ-s) ✓ EQUIVALENT
    - dist_sq term:
        Standard: dist_sq * (1-β_std)^(λ-s-1)
        Lazy:     dist_sq * 1

        These are NOT equivalent!

The lazy version does NOT apply decay to newly added dist_sq.
This causes errors to accumulate faster, leading to:
    - Different node selection for insertion
    - Different topology evolution
    - Different final edge counts
""")


if __name__ == "__main__":
    test_single_win()
    test_multiple_wins()
    test_no_wins()
    analyze_formula_difference()
