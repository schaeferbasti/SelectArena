"""Build the leaderboard tables from a run's results and (optionally) push them to the Space.

    python scripts/run_benchmark.py eval                 # writes <run dir>/results.csv
    python scripts/run_extra_benchmark.py collect        # optional: stability / validity results
    python scripts/export_leaderboard.py --push          # tables -> leaderboard/data -> Space

The tables and the main figure (Elo with stability and validity, as ``main_figure.png`` and
``main_figure.pdf``) land in ``leaderboard/data/`` (the folder the Space app reads). ``--push`` uploads that
folder to the Hugging Face Space (``SelectArena/Leaderboard`` by default) with ``huggingface_hub``;
log in first with ``hf auth login`` using a token that can write to the organization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from selectarena.config import load_benchmark_config, load_cluster_config
from selectarena.evaluation.leaderboard import VALIDITY_NOISE_MAIN_FIGURE, build_leaderboard, write_leaderboard
from selectarena.evaluation.plot import plot_main_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPACE = "SelectArena/Leaderboard"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--cluster-config", default=None)
    parser.add_argument("--results", default=None, help="results.csv (default: <run dir>/results.csv).")
    parser.add_argument("--extra-dir", default=None, help="Directory with stability/validity_results.csv.")
    parser.add_argument("--out", default=str(REPO_ROOT / "leaderboard" / "data"))
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for the Elo intervals.")
    parser.add_argument(
        "--validity-noise",
        type=float,
        default=VALIDITY_NOISE_MAIN_FIGURE,
        help="Noise level whose validity shades the main figure (negative: mean over noise levels).",
    )
    parser.add_argument("--push", action="store_true", help="Upload the tables to the Hugging Face Space.")
    parser.add_argument("--space", default=DEFAULT_SPACE, help="Space repo id for --push.")
    args = parser.parse_args()

    benchmark = load_benchmark_config(args.config)
    if args.results is None or args.extra_dir is None:
        run_dir = load_cluster_config(args.cluster_config).output_dir / benchmark.benchmark_name
        results_path = Path(args.results) if args.results else run_dir / "results.csv"
        extra_dir = Path(args.extra_dir) if args.extra_dir else run_dir / benchmark.extra_benchmark.results_subdir
    else:
        results_path, extra_dir = Path(args.results), Path(args.extra_dir)

    def _read_optional(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    tables = build_leaderboard(
        pd.read_csv(results_path),
        stability=_read_optional(extra_dir / "stability_results.csv"),
        validity=_read_optional(extra_dir / "validity_results.csv"),
        benchmark_name=benchmark.benchmark_name,
        n_bootstrap=args.n_bootstrap,
        validity_noise=args.validity_noise if args.validity_noise >= 0 else None,
    )
    paths = write_leaderboard(tables, args.out)
    figure_paths = [Path(args.out) / "main_figure.png", Path(args.out) / "main_figure.pdf"]
    plot_main_figure(tables["main_figure"], figure_paths)
    paths += figure_paths
    print(f"Wrote {len(paths)} files to {args.out}")

    if args.push:
        from huggingface_hub import upload_folder

        upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=args.out,
            path_in_repo="data",
            commit_message=f"Update leaderboard ({benchmark.benchmark_name})",
        )
        print(f"Pushed to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
