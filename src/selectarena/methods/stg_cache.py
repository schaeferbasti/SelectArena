"""Content-addressed STG selections shared across downstream prediction models."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def selection_key(selector, X, y, time_limit):
    """Hash training values/schema/order, labels, split seed, code and search settings.

    Device and downstream model are intentionally absent: a selection is computed
    once by the producer and then applied unchanged by every prediction model.
    No outer validation/test inputs or outcomes enter the key or the fit.
    """
    import pandas as pd

    settings = {name: getattr(selector, name) for name in selector.CACHE_SETTINGS}
    settings.update(
        problem_type=selector.problem_type,
        random_state=selector.random_state,
        encoding_permutation_seed=selector.encoding_permutation_seed,
        max_features=selector.max_features,
        time_limit=time_limit,
    )
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for name in ("stg.py", "stg_cache.py", "abstract.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for data in (X, y.to_frame()):
        schema = [
            (
                str(c),
                str(data[c].dtype),
                repr(getattr(data[c].dtype, "categories", None)),
                getattr(data[c].dtype, "ordered", None),
            )
            for c in data.columns
        ]
        digest.update(repr(schema).encode())
        digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def cached_selection(selector, X, y, time_limit, compute):
    """Publish atomic JSON under a cross-process lock; consumers never retrain."""
    cache_dir = os.environ.get("SELECTARENA_STG_CACHE_DIR")
    if not cache_dir:
        selector.selection_cache_ = {"enabled": False, "hit": False}
        return compute(X=X, y=y, time_limit=time_limit)
    from filelock import FileLock

    started = time.monotonic()
    key = selection_key(selector, X, y, time_limit)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    role = os.environ.get("SELECTARENA_STG_CACHE_ROLE", "auto")
    if role not in ("auto", "consumer"):
        raise ValueError("SELECTARENA_STG_CACHE_ROLE must be auto or consumer")

    def load():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["key"] != key or len(record["scores"]) < selector.max_features:
            raise ValueError(f"Invalid STG selection cache: {path}")
        for name, value in record["state"].items():
            setattr(selector, name, value)
        selector.selection_cache_ = {
            "enabled": True,
            "hit": True,
            "key": key,
            "source_fit_seconds": record["fit_seconds"],
            "lookup_wait_seconds": time.monotonic() - started,
        }
        selector._log(20, f"STG selection cache hit: {key}; reusing fitted gates without retraining")
        return dict(record["scores"])

    if role == "consumer":
        # Wait outside STG's compute budget for the GPU producer of this same key.
        wait_until = started + 2 * (time_limit or 3600) + 300
        while not path.exists():
            if time.monotonic() >= wait_until:
                raise TimeoutError(f"No STG producer result for training fingerprint {key}")
            time.sleep(0.25)
        return load()
    with FileLock(str(path) + ".lock", timeout=2 * (time_limit or 3600) + 300):
        if path.exists():
            return load()
        selector.selection_cache_ = {"enabled": True, "hit": False, "key": key}
        fit_start = time.monotonic()
        scores = compute(X=X, y=y, time_limit=time_limit)
        if len(scores) < selector.max_features:
            raise RuntimeError("STG training produced no usable gate ranking; refusing random padding")
        duration = time.monotonic() - fit_start
        state = {name: getattr(selector, name) for name in selector.CACHE_STATE}
        # Trial histories/weights are not needed to apply a cached selection.
        state["search_results_"] = selector.get_selection_diagnostics()["trials"]
        record = {"key": key, "fit_seconds": duration, "scores": list(scores.items()), "state": state}
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        selector.selection_cache_["source_fit_seconds"] = duration
        return scores
