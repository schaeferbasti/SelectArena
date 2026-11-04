"""Regenerate the SelectArena leaderboard website artifacts.

The publishing pipeline behind the leaderboard Space (https://huggingface.co/spaces/SelectArena/Leaderboard),
mirroring TabArena's ``scripts/run_generate_website_artifacts.py``. The Space's code lives in the
Space repo itself; this repo only produces the data it serves. The ``update-leaderboard`` skill
(``.claude/skills/update-leaderboard/``) runs the whole procedure below.

    python scripts/run_benchmark.py eval                  # <run dir>/results.csv
    python scripts/run_extra_benchmark.py collect         # optional: stability / validity results
    python scripts/run_generate_website_artifacts.py      # -> generated_website_artifacts/

Output (``--out``, default ``generated_website_artifacts/``, git-ignored):

- ``raw_website_artifacts/``: the paper's main figure (``main_figure.png`` / ``.pdf``) and the
  per-comparison scores. Not published.
- ``clean_website_artifacts/website_data/``: the Space's ``data/`` folder (CSV + self-contained
  HTML, no binaries), and ``clean_website_artifacts.zip`` with the same content.

Your method on the leaderboard, locally (no access to the published results needed):

    python scripts/run_generate_website_artifacts.py --with-published-results \
        --results <run dir>/results.csv [--extra-dir <dir with stability/validity_results.csv>]
    python scripts/preview_leaderboard.py                 # http://localhost:8000

``--with-published-results`` downloads the published results of all methods
(``selectarena.published_results``, several GB, once) into ``results/`` if they are not there yet,
and adds your own results (``--results`` / ``--extra-dir``), which replace the published results of
the same method on the same task.

Publishing procedure (maintainers; or run the ``update-leaderboard`` skill):

1. Run this script (the Elo intervals take most of the time; ``--elo-bootstrap-rounds 1`` for a
   quick check whose intervals are meaningless).
2. Clone the Space: ``git clone https://huggingface.co/spaces/SelectArena/Leaderboard``
   (log in once with ``hf auth login --add-to-git-credential``).
3. Replace its ``data/``: delete the old ``data/`` content, then copy in the content of
   ``generated_website_artifacts/clean_website_artifacts/website_data/``.
4. In the Space's ``index.html``, bump ``selectarena-version`` and add a version history entry.
5. In the Space checkout: ``python smoke_test.py``, preview with ``python -m http.server 8000``
   (http://localhost:8000), then commit and ``git push``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selectarena.config import load_benchmark_config
from selectarena.evaluation.leaderboard import VALIDITY_NOISE_MAIN_FIGURE
from selectarena.published_results import RESULT_FILES, ensure_published_results, merge_results
from selectarena.website import generate_website_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The result files copied off the cluster or downloaded with --with-published-results (git-ignored).
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _read_optional(path: Path | None) -> pd.DataFrame | None:
    if path is not None and path.exists():
        print(f"Reading {path}")
        return pd.read_csv(path)
    if path is not None:
        print(f"Not found (skipped): {path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"), help="Benchmark config.")
    parser.add_argument(
        "--results",
        default=None,
        help="Performance results (default: results/performance_results.csv). With --with-published-results: "
        "your own results, added to the published ones.",
    )
    parser.add_argument(
        "--extra-dir",
        default=None,
        help="Folder with stability_results.csv / validity_results.csv (default: the folder of --results).",
    )
    parser.add_argument(
        "--with-published-results",
        action="store_true",
        help="Rank against the published results of all methods: downloads them into results/ if missing "
        "(several GB, once) and adds --results / --extra-dir to them.",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "generated_website_artifacts"), help="Output folder.")
    parser.add_argument("--elo-bootstrap-rounds", type=int, default=1000, help="Bootstrap resamples per Elo interval.")
    parser.add_argument(
        "--validity-noise",
        type=float,
        default=VALIDITY_NOISE_MAIN_FIGURE,
        help="Noise level whose validity the leaderboard shows (negative: mean over noise levels).",
    )
    parser.add_argument("--no-paper-figures", action="store_true", help="Skip the matplotlib PNG/PDF figure.")
    parser.add_argument("--no-zip", action="store_true", help="Skip clean_website_artifacts.zip.")
    parser.add_argument(
        "--include-ablations", action="store_true", help="Rank ablation runs (e.g. 'MI [nofallback]') as methods."
    )
    parser.add_argument(
        "--include-unregistered",
        action="store_true",
        help="Rank methods that are not in the registry (e.g. retired fork-era methods); left out by default.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark_config(args.config)
    if args.with_published_results:
        published = ensure_published_results(DEFAULT_RESULTS_DIR)
        # Your own results: --results, and the extra results in --extra-dir (default: next to --results).
        own = dict.fromkeys(RESULT_FILES)
        if args.results:
            own["performance"] = Path(args.results)
        own_extra_dir = Path(args.extra_dir) if args.extra_dir else (own["performance"] or Path()).parent
        if args.results or args.extra_dir:
            own["stability"] = own_extra_dir / RESULT_FILES["stability"]
            own["validity"] = own_extra_dir / RESULT_FILES["validity"]
        frames = {}
        for kind in RESULT_FILES:
            mine = own[kind] if own[kind] is not None and own[kind].resolve() != published[kind].resolve() else None
            frames[kind] = merge_results(_read_optional(published[kind]), _read_optional(mine), kind)
        if frames["performance"] is None:
            parser.error(f"no performance results in {published['performance']}")
    else:
        if args.results is None:
            # The result files copied off the cluster, in results/.
            results_path = DEFAULT_RESULTS_DIR / RESULT_FILES["performance"]
            default_extra_dir = DEFAULT_RESULTS_DIR
        else:
            # A results.csv copied off the cluster: look for the extra results next to it.
            results_path = Path(args.results)
            default_extra_dir = results_path.parent
        extra_dir = Path(args.extra_dir) if args.extra_dir else default_extra_dir
        print(f"Reading {results_path}")
        frames = {
            "performance": pd.read_csv(results_path),
            "stability": _read_optional(extra_dir / RESULT_FILES["stability"]),
            "validity": _read_optional(extra_dir / RESULT_FILES["validity"]),
        }

    artifacts = generate_website_artifacts(
        frames["performance"],
        out_dir=args.out,
        stability=frames["stability"],
        validity=frames["validity"],
        benchmark_name=benchmark.benchmark_name,
        n_bootstrap=args.elo_bootstrap_rounds,
        validity_noise=args.validity_noise if args.validity_noise >= 0 else None,
        paper_figures=not args.no_paper_figures,
        zip_clean=not args.no_zip,
        include_ablations=args.include_ablations,
        include_unregistered=args.include_unregistered,
    )
    print(f"\n{len(artifacts.subsets)} subsets written to {artifacts.website_data}")
    if artifacts.zip_path:
        print(f"Zipped: {artifacts.zip_path}")
    print(f"Paper figure and raw tables: {artifacts.raw_dir}")
    if artifacts.unclassified_methods:
        print(f"WARNING: shown as 'Other' on the leaderboard: {artifacts.unclassified_methods}")
    print("To see the whole leaderboard with this data: python scripts/preview_leaderboard.py")


if __name__ == "__main__":
    main()
