"""The published SelectArena results, to rank a new method against the published ones.

The leaderboard is computed from the raw results of every method (the Elo compares methods on the
same dataset splits), so a new method can only be ranked against the published methods together
with their results. Those are the Hugging Face dataset repository :data:`RESULTS_REPO_ID`, with one
folder per method, named as on the leaderboard: ``methods/SHAP/performance_SHAP.csv`` (the downstream
errors), ``stability_SHAP.csv`` and ``validity_SHAP.csv`` (the selections of the extra benchmark).
Publishing a method means uploading its folder; a rerun replaces the folder.

:func:`ensure_published_results` syncs those folders into ``results/published/`` (git-ignored;
only new or changed files are downloaded), :func:`read_published_results` reads them, and
:func:`merge_results` adds a user's own results, which replace the published results of the same
method on the same task (e.g. a rerun). :func:`write_method_folders` sorts result files into method
folders, for an upload (``scripts/stage_published_results.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

RESULTS_REPO_ID = "SelectArena/SelectArena_Results"
#: Commit of :data:`RESULTS_REPO_ID` the results are read from. ``None`` reads the latest version;
#: pin it to a commit (as ``selectarena.datasets.HF_REVISION``) so a leaderboard is reproducible.
RESULTS_REVISION: str | None = None
#: Kind of result -> file name, locally and in :data:`RESULTS_REPO_ID`.
RESULT_FILES = {
    "performance": "performance_results.csv",
    "stability": "stability_results.csv",
    "validity": "validity_results.csv",
}
#: Folder of the method folders, in :data:`RESULTS_REPO_ID` and in the local copy.
METHODS_DIR = "methods"
#: The local copy of :data:`RESULTS_REPO_ID`, under the results folder.
PUBLISHED_SUBDIR = "published"
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
) -> Path:
    """Sync the published method folders into ``<results_dir>/published/`` and return that folder.

    Only new or changed files are downloaded. Without a connection, an existing local copy is used.
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(results_dir) / PUBLISHED_SUBDIR
    log(f"Syncing the published results from huggingface.co/datasets/{RESULTS_REPO_ID} into {local_dir}")
    try:
        snapshot_download(
            repo_id=RESULTS_REPO_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{METHODS_DIR}/*/*.csv"],
            local_dir=local_dir,
        )
    except OSError as error:  # the hub's connection and offline errors
        if not any((local_dir / METHODS_DIR).glob("*/*.csv")):
            raise
        log(f"  could not sync ({type(error).__name__}); using the local copy")
    return local_dir


def read_published_results(published_dir: str | Path) -> dict[str, pd.DataFrame | None]:
    """The results of all method folders in ``published_dir``, one table per kind (``None`` if none)."""
    frames = {}
    for kind in RESULT_FILES:
        files = sorted((Path(published_dir) / METHODS_DIR).glob(f"*/{kind}_*.csv"))
        frames[kind] = (
            pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True) if files else None
        )
    return frames


def detect_kind(path: str | Path) -> str:
    """``performance``, ``stability`` or ``validity``, from the columns of a result file (not its name)."""
    head = pd.read_csv(path, nrows=50)
    if "experiment_method_name_string" in head.columns:
        return "performance"
    modes = set(head["mode"].dropna().astype(str)) if "mode" in head.columns else set()
    if len(modes) == 1 and next(iter(modes)) in ("stability", "validity"):
        return next(iter(modes))
    raise ValueError(f"{path}: not a performance, stability or validity result file")


def method_folder(method_class: str) -> str:
    """The results folder of a feature selector class: its leaderboard name (old class names map to the current one)."""
    from selectarena.methods.registry import LEGACY_NAMES, NAME_TO_MODULE_MAP, PAPER_NAMES

    name = LEGACY_NAMES.get(method_class, method_class)
    if name in NAME_TO_MODULE_MAP:
        return PAPER_NAMES[name]
    raise KeyError(f"{method_class} is not registered (methods/registry.py), so it has no results folder")


def write_method_folders(
    paths: list[str | Path], out_dir: str | Path, *, chunksize: int = 500_000, log: Callable[[str], None] = print
) -> list[Path]:
    """Sort the rows of result files into ``<out_dir>/methods/<method>/<kind>_<method>.csv``.

    A file replaces what an earlier file (or an earlier call) wrote for the same method folder and kind, so
    pass a rerun after the results it replaces. Returns the files written.
    """
    written: dict[Path, int] = {}
    for path in map(Path, paths):
        kind = detect_kind(path)
        column = "experiment_method_name_string" if kind == "performance" else "method"
        classes = method_classes(pd.read_csv(path, usecols=[column]), kind)
        if classes.isna().any():
            raise ValueError(f"{path}: {int(classes.isna().sum())} rows without a FSBench__<Selector> pipeline")
        folders = {cls: method_folder(cls) for cls in classes.unique()}
        targets = {cls: Path(out_dir) / METHODS_DIR / f / f"{kind}_{f}.csv" for cls, f in folders.items()}
        for target in set(targets.values()):
            target.unlink(missing_ok=True)
            written[target] = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            row_targets = method_classes(chunk, kind).map(targets)
            for target, rows in chunk.groupby(row_targets, sort=False):
                target.parent.mkdir(parents=True, exist_ok=True)
                rows.to_csv(target, mode="a", header=not target.exists(), index=False)
                written[target] += len(rows)
        log(
            f"{path.name} ({kind}): "
            + ", ".join(f"{t.parent.name} {written[t]}" for t in sorted(set(targets.values())))
        )
    return sorted(written)


def merge_results(
    published: pd.DataFrame | None, own: pd.DataFrame | None, kind: str, *, replace_methods: bool = False
) -> pd.DataFrame | None:
    """The published results plus a user's own; own rows replace the published rows of the same
    method on the same task (see :data:`MERGE_KEYS`).

    With ``replace_methods``, own results replace *all* published results of their methods (every
    pipeline of the method, e.g. other time limits, and every task), for a method rerun from scratch
    whose published results are outdated.
    """
    if own is None or own.empty:
        return published
    if published is None or published.empty:
        return own
    if replace_methods:
        own_methods = set(method_classes(own, kind).dropna())
        published = published[~method_classes(published, kind).isin(own_methods)]
    keys = [k for k in MERGE_KEYS[kind] if k in published.columns and k in own.columns]
    if keys:
        replaced = pd.MultiIndex.from_frame(_merge_key_frame(own, keys)).unique()
        published = published[~pd.MultiIndex.from_frame(_merge_key_frame(published, keys)).isin(replaced)]
    return pd.concat([published, own], ignore_index=True, sort=False)


def method_classes(df: pd.DataFrame, kind: str) -> pd.Series:
    """The feature selector class of each result row (``NaN`` for rows without feature selection,
    such as the ``default`` pipeline), read from its ``FSBench__<Selector>__...`` pipeline."""
    column = "experiment_method_name_string" if kind == "performance" else "method"
    if column not in df:
        return pd.Series(pd.NA, index=df.index, dtype=object)
    return df[column].astype(str).str.extract(r"FSBench__(.+?)__", expand=False)


def _task_id_key(task_id: str) -> str:
    """A Data Foundry task id without the cache path some runs append to it:
    ``UserTask|<id>|<dataset>/<uuid>|/work/.../tabarena_tasks`` -> ``UserTask|<id>|<dataset>/<uuid>``,
    so results of the same task match whether or not they carry the path."""
    return "|".join(task_id.split("|")[:3]) if task_id.startswith("UserTask|") else task_id


def _merge_key_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame = df[keys].astype(str)
    if "data_foundry_task_id" in frame:
        frame["data_foundry_task_id"] = frame["data_foundry_task_id"].map(_task_id_key)
    return frame
