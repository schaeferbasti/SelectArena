"""Check feature selection methods against the benchmark's contract (see selectarena.methods.checks).

    python scripts/check_method.py MyNewFeatureSelector        # one method (its class name)
    python scripts/check_method.py --all                       # every registered method
    python scripts/check_method.py MyNewFeatureSelector --no-time-limit

Checks the registration (registry, paper name, taxonomy subgroup, proxy-model listing), fits the
method on small binary, multiclass and regression data (output format, budget, determinism) and
checks that it stops near its time limit on a larger dataset. Exits with 1 if a check fails.
"""

from __future__ import annotations

import argparse
import sys
import warnings

from selectarena.methods.checks import check_method
from selectarena.methods.registry import FEATURE_SELECTION_METHODS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("methods", nargs="*", help="Class names of the methods to check.")
    parser.add_argument("--all", action="store_true", help="Check every registered method.")
    parser.add_argument("--no-fit", action="store_true", help="Only the static checks (registry, naming).")
    parser.add_argument("--no-time-limit", action="store_true", help="Skip the time-limit check.")
    args = parser.parse_args()
    warnings.simplefilter("ignore")  # sklearn / AutoGluon warnings on the tiny check datasets
    names = FEATURE_SELECTION_METHODS if args.all else args.methods
    if not names:
        parser.error("name at least one method or pass --all")

    failed = []
    for name in names:
        print(f"\n== {name}")
        results = check_method(name, fit=not args.no_fit, time_limit_check=not (args.no_fit or args.no_time_limit))
        for result in results:
            print(result)
        if not all(r.ok for r in results):
            failed.append(name)
    print(f"\n{len(names) - len(failed)}/{len(names)} methods pass" + (f"; failing: {failed}" if failed else "."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
