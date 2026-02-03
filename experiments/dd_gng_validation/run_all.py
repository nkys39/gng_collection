"""Run all DD-GNG validation tests.

Usage:
    python run_all.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("DD-GNG Implementation Validation")
    print("Based on: Saputra et al. (2019)")
    print("=" * 70)

    results = {}

    # Test 1: Strength Calculation
    print("\n" + "=" * 70)
    print("Running Test 1: Strength Calculation...")
    print("=" * 70)
    from test_strength import main as test_strength
    results["strength"] = test_strength()

    # Test 2: Density Comparison
    print("\n" + "=" * 70)
    print("Running Test 2: Density Comparison...")
    print("=" * 70)
    from test_density import main as test_density
    results["density"] = test_density()

    # Test 3: Ladder Detection
    print("\n" + "=" * 70)
    print("Running Test 3: Ladder Detection...")
    print("=" * 70)
    from test_ladder import main as test_ladder
    results["ladder"] = test_ladder()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTest 1 (Strength Calculation):  {'PASS' if results['strength'] else 'FAIL'}")
    print(f"Test 2 (Density Comparison):    {'PASS' if results['density'] else 'FAIL'}")
    print(f"Test 3 (Ladder Detection):      {'PASS' if results['ladder'] else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    print("\nOutput files saved to: experiments/dd_gng_validation/outputs/")
    print("  - strength_test.gif / strength_test.png")
    print("  - density_comparison.gif / density_comparison.png")
    print("  - ladder_detection.gif / ladder_detection.png")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
