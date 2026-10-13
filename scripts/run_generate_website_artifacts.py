"""Maintainer-only: regenerate the SelectArena leaderboard website artifacts.

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

Publishing procedure (or run the ``update-leaderboard`` skill):

1. Run this script (the Elo intervals take most of the time; ``--elo-bootstrap-rounds 1`` for a
   quick check whose intervals are meaningless).
2. Clone the Space: ``git clone https://huggingface.co/spaces/SelectArena/Leaderboard``
   (log in once with ``hf auth login --add-to-git-credential``).
3. Replace its ``data/``: delete the old ``data/`` content, then copy in the content of
   ``generated_website_artifacts/clean_website_artifacts/website_data/``.
4. Add a version history entry in the Space's ``website_texts.py`` and bump ``version`` in its
   ``pyproject.toml``.
5. Preview with ``python main.py`` in the Space checkout, then commit and ``git push``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selectarena.config import load_benchmark_config, load_cluster_config
from selectarena.evaluation.leaderboard import VALIDITY_NOISE_MAIN_FIGURE
from selectarena.website import generate_website_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"), help="Benchmark config.")
    parser.add_argument(
        "--cluster-config", default=None, help="Cluster config, to find the run directory when --results is not given."
    )
    parser.add_argument("--results", default=None, help="results.csv (default: <run dir>/results.csv).")
    parser.add_argument(
        "--extra-dir",
        default=None,
        help="Folder with stability_results.csv / validity_results.csv (default: <run dir>/<results_subdir>).",
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
    if args.results is None:
        run_dir = load_cluster_config(args.cluster_config).output_dir / benchmark.benchmark_name
        results_path = run_dir / "results.csv"
    else:
        # A results.csv copied off the cluster: look for the extra results next to it.
        results_path = Path(args.results)
        run_dir = results_path.parent
    extra_dir = Path(args.extra_dir) if args.extra_dir else run_dir / benchmark.extra_benchmark.results_subdir

    def _read_optional(path: Path) -> pd.DataFrame | None:
        if path.exists():
            print(f"Reading {path}")
            return pd.read_csv(path)
        print(f"Not found (skipped): {path}")
        return None

    print(f"Reading {results_path}")
    artifacts = generate_website_artifacts(
        pd.read_csv(results_path),
        out_dir=args.out,
        stability=_read_optional(extra_dir / "stability_results.csv"),
        validity=_read_optional(extra_dir / "validity_results.csv"),
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


if __name__ == "__main__":
    main()
