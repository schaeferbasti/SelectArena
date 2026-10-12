"""Publish the SelectArena dataset suite on Hugging Face, the way TabArena publishes BeyondArena.

The Hub dataset repo mirrors the Data Foundry warehouse layout: every curated container sits at
``<dataset>/<uuid>/`` from the repo root (``dataset.parquet``, ``dtypes.json``, the three
``*.json`` metadata files, ``container_metadata.json``, plus any extra files). Data Foundry's
``HuggingFaceSource`` reads exactly that layout, so the collection in :mod:`selectarena.datasets`
can download single datasets on demand.

Next to the containers the repo root gets a dataset card (``README.md``), a table of the datasets
(``datasets.csv``) and the original datasets' references (``CITATIONS.bib``), all generated from
the containers' own metadata. Reading the metadata needs no Data Foundry install: it is plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from selectarena.datasets import BENCHMARK_DATA_FOUNDRY_TASKS, HF_REPO_ID

CORE_FILES = ("dataset.parquet", "dtypes.json", "container_metadata.json")
METADATA_FILES = {
    "dataset": "dataset_metadata.dataset-mold-v1.json",
    "task": "task_metadata.predictive-ml-task-mold-v1.json",
}

PAPER_BIBTEX = """@article{schaefer2026selectarena,
  title   = {SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data},
  author  = {Sch{\\"a}fer, Bastian and Matus, Dominika and Purucker, Lennart and Knauer, Ricardo and
             Rodner, Erik and Hutter, Frank and Sokolov, Anton and Liefooghe, Arnaud},
  journal = {arXiv preprint arXiv:2610.13644},
  year    = {2026}
}"""


@dataclass
class ContainerInfo:
    """What the dataset card needs to know about one container."""

    relative_path: str
    exists: bool
    missing_files: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.relative_path.split("/")[0]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _parquet_shape(path: Path) -> tuple[int | None, int | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, None
    meta = pq.read_metadata(path)
    return meta.num_rows, meta.num_columns


def inspect_container(warehouse: Path, relative_path: str) -> ContainerInfo:
    """Check a container's files and read its dataset/task metadata."""
    path = warehouse / relative_path
    if not path.is_dir():
        return ContainerInfo(relative_path, exists=False)
    expected = [*CORE_FILES, *METADATA_FILES.values()]
    info = ContainerInfo(relative_path, exists=True, missing_files=[f for f in expected if not (path / f).is_file()])
    dataset = _read_json(path / METADATA_FILES["dataset"]) if (path / METADATA_FILES["dataset"]).is_file() else {}
    task = _read_json(path / METADATA_FILES["task"]) if (path / METADATA_FILES["task"]).is_file() else {}
    n_rows, n_columns = (
        _parquet_shape(path / "dataset.parquet") if (path / "dataset.parquet").is_file() else (None, None)
    )
    info.metadata = {
        "dataset": info.name,
        "uuid": relative_path.split("/")[-1],
        "path": relative_path,
        "problem_type": task.get("problem_type"),
        "target": task.get("target_column_name"),
        "metric": task.get("objective_metric_name"),
        "n_rows": n_rows,
        "n_features": n_columns - 1 if n_columns else None,
        "domain": dataset.get("domain_str"),
        "year": dataset.get("dataset_year"),
        "source": dataset.get("dataset_source"),
        "source_link": dataset.get("original_dataset_source_download_link"),
        "license": dataset.get("license"),
        "bibtex_key": dataset.get("academic_reference_bibtex_key"),
        "bibtex": dataset.get("academic_reference_bibtex"),
    }
    return info


def inspect_suite(warehouse: str | Path, relative_paths: list[str] | None = None) -> list[ContainerInfo]:
    """Inspect every container of the suite in a local warehouse."""
    warehouse = Path(warehouse).expanduser()
    return [inspect_container(warehouse, p) for p in (relative_paths or BENCHMARK_DATA_FOUNDRY_TASKS)]


def datasets_table(infos: list[ContainerInfo]) -> pd.DataFrame:
    rows = [{k: v for k, v in i.metadata.items() if k != "bibtex"} for i in infos if i.exists]
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def citations(infos: list[ContainerInfo]) -> str:
    """The original datasets' BibTeX entries, deduplicated."""
    entries = {i.metadata.get("bibtex", "").strip() for i in infos if i.exists}
    return "\n\n".join(sorted(e for e in entries if e)) + "\n"


def _size_category(n: int) -> str:
    return "n<1K" if n < 1_000 else "1K<n<10K" if n < 10_000 else "10K<n<100K" if n < 100_000 else "100K<n<1M"


def dataset_card(table: pd.DataFrame, *, repo_id: str = HF_REPO_ID, excluded: list[str] | None = None) -> str:
    """The Hub dataset card (``README.md``) for the uploaded suite."""
    task_categories = sorted(
        {
            "tabular-regression" if p == "regression" else "tabular-classification"
            for p in table["problem_type"].dropna()
        }
    )
    max_rows = int(table["n_rows"].max()) if table["n_rows"].notna().any() else 0
    header = [
        "---",
        "pretty_name: SelectArena",
        "license: other",
        "license_name: per-dataset",
        "license_link: LICENSE.md",
        "task_categories:",
        *[f"- {c}" for c in task_categories],
        "tags:",
        "- feature-selection",
        "- tabular",
        "- benchmark",
        "- data-foundry",
        "size_categories:",
        f"- {_size_category(max_rows)}",
        "---",
    ]
    columns = ["dataset", "problem_type", "n_rows", "n_features", "metric", "domain", "year", "license", "source_link"]
    rows = []
    for _, r in table[columns].iterrows():
        cells = ["" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in r]
        cells[-1] = f"[link]({cells[-1]})" if cells[-1] else ""
        rows.append("| " + " | ".join(cells) + " |")
    excluded_text = ""
    if excluded:
        excluded_text = (
            "\n### Datasets not hosted here\n\n"
            "These datasets are part of the benchmark but their licence does not allow redistribution. "
            "Obtain them from their original source (see the paper's appendix):\n\n"
            + "\n".join(f"- `{name}`" for name in excluded)
            + "\n"
        )
    by_type = table["problem_type"].value_counts().to_dict()
    body = f"""
# SelectArena datasets

The real-world tabular datasets of [SelectArena](https://github.com/schaeferbasti/SelectArena), a benchmark for
feature selection methods ([paper](https://arxiv.org/abs/2610.13644)). The datasets were curated with
[Data Foundry](https://github.com/TabArena/data-foundry), the curation tooling behind TabArena's BeyondArena.

{len(table)} datasets are hosted here: {", ".join(f"{n} {t.replace('_', ' ')}" for t, n in sorted(by_type.items()))}.

## Layout

Every dataset is a Data Foundry curated container at `<dataset>/<uuid>/`:

- `dataset.parquet` and `dtypes.json`: the data and its column types;
- `dataset_metadata.dataset-mold-v1.json`: source, licence, reference, domain;
- `task_metadata.predictive-ml-task-mold-v1.json`: target, problem type, evaluation metric;
- `experiment_metadata.predictive-ml-splits-mold-v1.json`: the outer cross-validation splits;
- `container_metadata.json`: container id and checksum.

`datasets.csv` lists all datasets with their metadata; `CITATIONS.bib` holds the original references.

## Usage

With SelectArena (downloads only the datasets a run needs):

```python
from selectarena.datasets import make_collection

collection = make_collection()            # reads from {repo_id}
container = collection.get_dataset("{table["dataset"].iloc[0] if len(table) else "anneal"}")
print(container.dataset.head())
```

With Data Foundry directly:

```python
from data_foundry.collections import DatasetCollection, HuggingFaceSource

collection = DatasetCollection.from_relative_paths(
    name="SelectArena",
    description="SelectArena datasets",
    relative_paths=[...],                 # the `path` column of datasets.csv
    source=HuggingFaceSource(repo_id="{repo_id}"),
)
```

## Datasets

| Dataset | Problem type | Rows | Features | Metric | Domain | Year | Licence | Source |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}
{excluded_text}
## Licences

Each dataset keeps the licence of its original source, listed in the table above and in each container's
`dataset_metadata` file. When you use a dataset, cite its original reference (`CITATIONS.bib`) as well as
SelectArena.

## Citation

```bibtex
{PAPER_BIBTEX}
```
"""
    return "\n".join(header) + "\n" + body


def license_md(table: pd.DataFrame) -> str:
    lines = ["# Licences", "", "Each dataset is distributed under the licence of its original source:", ""]
    lines += [f"- `{r.dataset}`: {r.license or 'unknown'} ({r.source_link})" for r in table.itertuples()]
    return "\n".join(lines) + "\n"


def write_repo_files(infos: list[ContainerInfo], out_dir: str | Path, **card_kwargs) -> list[Path]:
    """Write ``README.md``, ``LICENSE.md``, ``datasets.csv`` and ``CITATIONS.bib`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = datasets_table(infos)
    files = {
        "README.md": dataset_card(table, **card_kwargs),
        "LICENSE.md": license_md(table),
        "CITATIONS.bib": citations(infos),
    }
    paths = []
    for name, text in files.items():
        (out_dir / name).write_text(text)
        paths.append(out_dir / name)
    table.to_csv(out_dir / "datasets.csv", index=False)
    return [*paths, out_dir / "datasets.csv"]
