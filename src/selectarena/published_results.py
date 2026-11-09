"""The published SelectArena results, to rank a new method against the published ones.

The leaderboard is computed from the raw results of every method (the Elo compares methods on the
same dataset splits), so a new method can only be ranked against the published methods together
with their results. Those are the Hugging Face dataset repository :data:`RESULTS_REPO_ID`:
``performance_results.csv`` (the downstream errors), ``stability_results.csv`` and
``validity_results.csv`` (the selections of the extra benchmark), at its root.

:func:`ensure_published_results` downloads the files that are not in the local results folder yet
(``results/`` of the repo, git-ignored, where ``scripts/run_generate_website_artifacts.py`` reads
them); :func:`merge_results` adds a user's own results, which replace the published results of the
same method on the same task (e.g. a rerun).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

RESULTS_REPO_ID = "ANONYMIZED/SelectArena_Results"
#: Commit of :data:`RESULTS_REPO_ID` the results are read from. ``None`` reads the latest version;
#: pin it to a commit (as ``selectarena.datasets.HF_REVISION``) so a leaderboard is reproducible.
RESULTS_REVISION: str | None = None
#: Kind of result -> file name, locally and in :data:`RESULTS_REPO_ID`.
RESULT_FILES = {
    "performance": "performance_results.csv",
    "stability": "stability_results.csv",
    "validity": "validity_results.csv",
}
#: Columns that identify a method's results on one task, per kind of result: own results replace
#: the published ones with the same values.
MERGE_KEYS = {
    "performance": ["experiment_method_name_string", "name"],
    "stability": ["method", "data_foundry_task_id"],
    "validity": ["method", "data_foundry_task_id"],
}


def ensure_published_results(
    results_dir: str | Path,
    *,
    revision: str | None = RESULTS_REVISION,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Return the local paths of the published result files (by kind), downloading the missing ones.

    A file already in ``results_dir`` is used as it is (delete it to download it again).
    """
    results_dir = Path(results_dir)
    paths = {kind: results_dir / name for kind, name in RESULT_FILES.items()}
    missing = [kind for kind, path in paths.items() if not path.is_file()]
    if missing:
        from huggingface_hub import hf_hub_download

        results_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"Downloading the published {', '.join(missing)} results from huggingface.co/datasets/{RESULTS_REPO_ID} "
            f"into {results_dir} (several GB in total; only once)."
        )
        for kind in missing:
            hf_hub_download(
                repo_id=RESULTS_REPO_ID,
                filename=RESULT_FILES[kind],
                repo_type="dataset",
                revision=revision,
                local_dir=results_dir,
            )
            log(f"  {paths[kind]}")
    return paths


def merge_results(published: pd.DataFrame | None, own: pd.DataFrame | None, kind: str) -> pd.DataFrame | None:
    """The published results plus a user's own; own rows replace the published rows of the same
    method on the same task (see :data:`MERGE_KEYS`)."""
    if own is None or own.empty:
        return published
    if published is None or published.empty:
        return own
    keys = [k for k in MERGE_KEYS[kind] if k in published.columns and k in own.columns]
    if keys:
        replaced = pd.MultiIndex.from_frame(own[keys].astype(str)).unique()
        published = published[~pd.MultiIndex.from_frame(published[keys].astype(str)).isin(replaced)]
    return pd.concat([published, own], ignore_index=True, sort=False)
