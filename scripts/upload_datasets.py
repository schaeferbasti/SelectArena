"""Upload the SelectArena datasets to Hugging Face, mirroring the Data Foundry warehouse layout.

    # 1. Check the local containers and write the dataset card, nothing is uploaded:
    python scripts/upload_datasets.py --warehouse ~/Programming/ULCO/Autogluon/data_foundry/local_data_warehouse

    # 2. Review the printed licence report and data_release/README.md, then upload:
    hf auth login                      # token with write access to the SelectArena organization
    python scripts/upload_datasets.py --warehouse ... --upload [--exclude excluded.txt] [--private]

The Hub repo gets the 71 containers at ``<dataset>/<uuid>/`` (the layout Data Foundry's
``HuggingFaceSource`` reads, as for TabArena's BeyondArena) plus ``README.md`` (dataset card),
``LICENSE.md``, ``datasets.csv`` and ``CITATIONS.bib``. ``--exclude`` takes a file with one dataset
name per line (e.g. datasets whose licence forbids redistribution); they are left out of the upload
and listed in the card instead. Only the suite's containers are uploaded, never the rest of the
warehouse. The upload resumes where it stopped when rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from selectarena.data_release import datasets_table, inspect_suite, write_repo_files
from selectarena.datasets import BENCHMARK_DATA_FOUNDRY_TASKS, HF_REPO_ID

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_excludes(path: str | None) -> set[str]:
    if path is None:
        return set()
    lines = Path(path).read_text().splitlines()
    return {line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warehouse", required=True, help="Local Data Foundry warehouse (<dataset>/<uuid>/...).")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help=f"Hub dataset repo (default: {HF_REPO_ID}).")
    parser.add_argument("--exclude", default=None, help="File with dataset names to leave out, one per line.")
    parser.add_argument("--out", default=str(REPO_ROOT / "data_release"), help="Where to write the card files.")
    parser.add_argument("--upload", action="store_true", help="Upload to the Hub (default: only check and write).")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--num-workers", type=int, default=None, help="Upload workers (huggingface_hub default).")
    args = parser.parse_args()

    warehouse = Path(args.warehouse).expanduser().resolve()
    excludes = _read_excludes(args.exclude)
    unknown = excludes - {p.split("/")[0] for p in BENCHMARK_DATA_FOUNDRY_TASKS}
    if unknown:
        sys.exit(f"--exclude lists datasets that are not in the suite: {sorted(unknown)}")
    paths = [p for p in BENCHMARK_DATA_FOUNDRY_TASKS if p.split("/")[0] not in excludes]

    infos = inspect_suite(warehouse, paths)
    missing = [i.relative_path for i in infos if not i.exists]
    incomplete = {i.relative_path: i.missing_files for i in infos if i.exists and i.missing_files}
    if missing or incomplete:
        for p in missing:
            print(f"MISSING     {warehouse / p}")
        for p, files in incomplete.items():
            print(f"INCOMPLETE  {p}: no {', '.join(files)}")
        sys.exit(f"{len(missing)} containers missing, {len(incomplete)} incomplete. Nothing was uploaded.")

    table = datasets_table(infos)
    print(f"All {len(table)} containers found in {warehouse} ({len(excludes)} excluded).\n")
    print("Licences (check that each one allows redistribution):")
    for licence, group in table.fillna({"license": "UNKNOWN"}).groupby("license"):
        print(f"  {licence:<28} {len(group):>3}  {', '.join(group['dataset'])}")
    print()

    files = write_repo_files(infos, args.out, repo_id=args.repo_id, excluded=sorted(excludes))
    print(f"Wrote {', '.join(f.name for f in files)} to {args.out}")
    if not args.upload:
        print("\nDry run: review the files above, then rerun with --upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Uploading {len(paths)} containers to https://huggingface.co/datasets/{args.repo_id} ...")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(warehouse),
        allow_patterns=[f"{p}/*" for p in paths],
        ignore_patterns=["**/.DS_Store", "**/._*"],
        num_workers=args.num_workers,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.out,
        commit_message="Update dataset card, licences, dataset table and citations",
    )
    print("Done.")


if __name__ == "__main__":
    main()
