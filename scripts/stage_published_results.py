"""Sort result CSVs into the published layout (one folder per method), ready to upload.

Writes ``results/published/methods/<method>/<kind>_<method>.csv`` (e.g. ``methods/SHAP/performance_SHAP.csv``,
the leaderboard name): the local copy of the published results (``selectarena.published_results``), so the
leaderboard can use them before and after the upload. A file replaces what an earlier file wrote for the same method and kind: pass reruns last.

    python scripts/stage_published_results.py performance_deep_lasso_r0f0-r0f2.csv stability_deep_lasso.csv ...

Then upload the printed folders (maintainers; this publishes them), right away: the next sync of
``--with-published-results`` would restore a changed folder from Hugging Face.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from selectarena.published_results import METHODS_DIR, PUBLISHED_SUBDIR, RESULTS_REPO_ID, write_method_folders

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path, help="performance / stability / validity result files")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / PUBLISHED_SUBDIR, help="default: results/published"
    )
    args = parser.parse_args()

    written = write_method_folders(args.csv, args.out)
    print("\nTo publish (one command per method folder):")
    for folder in sorted({path.parent for path in written}):
        target = f"{METHODS_DIR}/{folder.name}"
        # --delete is relative to the target folder: a rerun's folder replaces the old one completely.
        command = ["hf", "upload", RESULTS_REPO_ID, str(folder), target, "--repo-type", "dataset", "--delete", "*"]
        print(shlex.join([*command, "--commit-message", f"Results of {folder.name}"]))


if __name__ == "__main__":
    main()
