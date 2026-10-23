"""The SelectArena dataset suite and how it becomes a TabArena task collection.

The suite is a list of Data Foundry containers (``<dataset>/<container uuid>``), published in the
Hugging Face dataset repository :data:`HF_REPO_ID`. They are downloaded and converted once into
TabArena ``UserTask``s with :func:`prepare_datasets` (writes the task pickles into the OpenML cache
and a ``<COLLECTION_NAME>_tasks_metadata.csv`` next to them); afterwards
:func:`make_task_metadata_collection` reads that CSV without touching Data Foundry again.

Requires ``tabarena[data-foundry]`` for the conversion step.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

COLLECTION_NAME = "feature_selection_benchmark"
COLLECTION_DESCRIPTION = "SelectArena: tabular datasets for benchmarking feature selection methods."

BENCHMARK_DATA_FOUNDRY_TASKS = [
    "allstate_claims_severity/019dab3b-bfcc-7f19-8def-ddc92f50f1fd",
    "ancestry_study/019d9cb7-888d-7a7f-9e2b-4b13a0facdcb",
    "anneal/019d9cb8-f8ea-7742-b8bd-29c5ceef0d12",
    "aps_failure/019d9cba-c1ec-7353-9b0f-3feaed867a83",
    "audiology_diagnosis/019d9cbe-7fc4-788e-bd47-f4023e92a32a",
    "automobile/019daacc-fe0f-74ea-a269-e2057dd651fb",
    "bad_customer_detection/019d9cc1-3116-7d64-81b1-87119aa1fbdb",
    "bank_customer_churn/019d9cc2-2db2-7592-b439-3e8aa1e1cd29",
    "bank_marketing/019d9cc3-2925-71e4-b7fd-976c8c779195",
    "bioresponse/019d9cc4-01fb-72e6-bb56-3df49d2be492",
    "body_density_prediction/019d9cc4-a901-7b17-8ea9-277e92b37579",
    "brasilian_houses/019d9cc6-513b-7c8a-ae7a-30daf2cc5ad2",
    "churn/019d9cd1-09df-75e4-a0b5-57923a0accc8",
    "clock_protein_period/019d9cd1-aa5a-7b4d-a6a3-75423c198efe",
    "clock_protein_toxicity/019d9cd3-5abe-7b34-92e6-44ce41c0b9d0",
    "coil_2000_insurance_policies/019d9cd5-876e-7cd5-876c-2edf17c699e9",
    "colon_tumor/019d9cd6-7112-7d0e-a09d-f0e137b371cf",
    "credit_approval/019d9cd7-5cc2-72b1-8dca-a5437746e1fd",
    "credit_card_clients_default/019d9cd8-30ff-7112-b4a8-8eb093c099e3",
    "customer_satisfaction_in_airline/019d9cd9-939e-7f38-ae13-1a2c3a6e3d5f",
    "drug_induced_autoimmunity_prediction/019d9cda-7830-7cbd-8970-03d0daa0b64f",
    "forest_fires/019d9cdb-6862-7bcb-8733-cb9db5f573ed",
    "framingham_heart_study/019d9cdc-9a0d-7c91-ba72-e2c31fd7c833",
    "give_me_some_credit/019d9cdd-4000-7593-9945-fae38e6aefeb",
    "hazelnut_spread_contaminant_detection_10GHz/019d9cde-0193-7306-935a-65c5e6e3358e",
    "heart_disease_cleveland/019d9cdf-175e-7644-911d-c70dfeeb60e1",
    "heart_disease_hungary/019d9ce0-2349-7e3f-ada9-a96dfad347a6",
    "heart_disease_switzerland/019d9cdf-3ce5-7e78-840a-3dffae2b965e",
    "heart_disease_va_long_beach/019d9cdf-4abd-7146-9aa4-44f9044c19fd",
    "heloc/019d9ce3-7a8d-733e-b73e-ec49ee5b4ba1",
    "hepatitis_survival_prediction/019d9ce4-8eba-7478-9a8a-9584b8f802d8",
    "hiva_agnostic/019d9ce5-6be1-7c15-9229-195b5fb16816",
    "homesite_quote_conversion/019d9ce5-e6f1-7db6-b396-47113188bb48",
    "horse_colic_survival/019d9ce7-0f70-7af9-8d83-ff749e7bd6d6",
    "hr_analytics/019d9ce7-8a62-7716-9d2d-b328f6f77b30",
    "in_vehicle_coupon_recommendation/019d9cee-0ec8-7884-9383-49bc5a67e629",
    "ionosphere/019d9cee-9ce6-7536-9c24-cba65142db79",
    "japanese_credit_screening/019d9cf0-01ae-78c4-98ac-e24ca7db636b",
    "kdd_cup_09_appetency/019d9cf0-6453-77a2-850d-625509c50f74",
    "labor_negotiations/019d9d0d-bf29-7add-ab60-f643f871cd21",
    "leukemia_allaml/019d9d0f-09f2-7aca-8924-fd2db0f15985",
    "ljubljana_primary_tumor/019d9d12-98c1-77c1-96f9-4019be010f83",
    "lung_cancer/019d9cb4-8cf9-7763-80bc-e3a7acd09205",
    "lymphography/019d9d17-2e58-7cff-9c5c-0fdf9a3eb337",
    "marketing_campaign/019dab02-f15d-7be6-bc14-82228c12184d",
    "mechanisms_of_action/019daaf5-1f26-7268-b590-e8f860981af3",
    "miami_housing/019d9d1b-918d-7306-8499-edb088fb5569",
    "mic/019d9d1c-5f21-709c-88bf-a2c41c3f9f91",
    "naticusdroid_android_permissions_dataset/019d9d1c-d73f-7651-9cb3-ccf7356fb99b",
    "nci_ovarian_cancer/019d9d1f-b718-7871-ae91-44ed77e79e46",
    "nci_pancreatic_cancer/019d9d23-4ace-71c9-b3c3-06491168b970",
    "nci_prostate_cancer/019d9d25-7536-7943-b1b1-deee060a4115",
    "online_shoppers_purchasing_intention_dataset/019d9d27-56dd-7972-80b6-ba4ad600e054",
    "otto_group_product_classification_challenge/019d9d2a-4b55-7d84-9b0e-31e666e1653a",
    "polish_companies_bankruptcy/019d9d2a-a55b-7465-b41d-7a7c3cc16d13",
    "porto_seguro/019d9d2c-5062-70cb-b1e2-862e7dcb2972",
    "predict_students_dropout_and_academic_success/019d9d2c-d03c-7327-b13f-e4edffddb4fa",
    "prostate_cancer/019d9d2f-dc39-73bc-939b-ab86687ff5db",
    "pva_revenue_prediction_kddcup98/019d9d32-8356-7378-96ad-ed61dad561e7",
    "qsar_biodeg/019d9d33-6887-765f-8a54-6ba2d35f51df",
    "qsar_oral_toxicity/019d9d33-cdb5-7c70-96f9-f921748b363a",
    "qsar_tid_11/019d9d34-2c3f-727d-a73f-b93090124212",
    "santander_customer_transaction_prediction/019d9d37-22b4-745f-87a6-d27d45e652ec",
    "sdss_17/019d9d37-9471-7961-8231-3136cd4b9ced",
    "smoking_lung_cancer/019dab1b-1716-7761-b444-e07705a617d1",
    "soybean_large/019d9d3d-0110-7687-9a23-62920e7c0394",
    "srbct_prediction/019d9d3f-6ece-7759-91ac-ca896133c417",
    "superconductivity/019d9d3f-ccf7-70ad-883e-ab3e48e259fa",
    "taiwanese_bankruptcy_prediction/019d9d40-3e41-7afc-873a-8e4ef598dc12",
    "thyroid_discordant/019d9d40-cdac-7367-827c-bf2b6cb570b2",
    "wine_quality/019d9d40-e358-7271-88e9-2277666b5a62",
]

#: Earlier curation of the same suite (Data Foundry ``datasets/_dev/feature_selection``), which the
#: runs of ``tabarena_fork_fs`` used. Only needed to name the datasets of results from those runs.
EARLIER_DATA_FOUNDRY_TASKS = [
    "allstate_claims_severity/019c0a71-9029-727e-a7d9-a4c48238c737",
    "ancestry_study/019d43fa-c81a-7988-86e7-2f32c4268d6a",
    "anneal/019d3f7b-494a-71fa-8eb2-25d01dfb7792",
    "aps_failure/019d4414-84f1-763d-ab04-d8faf9e31c47",
    "audiology_diagnosis/019c7be3-f167-7218-a684-e8cd760cf045",
    "automobile/019d4417-9185-7412-aaa7-ea599ad207d3",
    "bad_customer_detection/019d441e-1bae-7f12-aec3-f882a691ba70",
    "bank_customer_churn/019d442c-fae5-748f-b144-6bd4c791f187",
    "bank_marketing/019d442e-8349-701a-a955-66e7e722724f",
    "bioresponse/019d4431-f400-7589-84da-05e4a83d0cd2",
    "body_density_prediction/019c71b9-743a-7d8b-b130-14340882cf6d",
    "brasilian_houses/019d4437-1097-76ab-86c2-1c453af76050",
    "churn/019d4438-bdb0-78de-858d-041394b879ae",
    "clock_protein_period/019d443a-2314-752d-91a3-7922279ad7fd",
    "clock_protein_toxicity/019d443b-bfe5-7c51-8f37-adcbc9ba6f72",
    "coil_2000_insurance_policies/019d443d-cbac-718a-98ec-b3a9ae46e867",
    "colon_tumor/019d4440-181a-776d-b896-984903dc749b",
    "credit_approval/019cb05b-bb5e-736e-8b40-7f5528c6fe6f",
    "credit_card_clients_default/019d4444-27b6-7cd6-81b7-8d97dfdb7bca",
    "customer_satisfaction_in_airline/019d4444-a63c-7efa-9537-5108e14ef4c0",
    "drug_induced_autoimmunity_prediction/019c9f82-d8aa-7c13-b593-d70835a80e0a",
    "forest_fires/019c71ac-de50-7c3e-bd8d-84dec1f5213f",
    "framingham_heart_study/019d4448-90df-7e75-b057-b72b7edfd6a8",
    "give_me_some_credit/019d4448-cbe4-71ad-9236-a9fe6a291951",
    "hazelnut_spread_contaminant_detection_10GHz/019d4452-ccab-7144-b341-ab8a2375aa88",
    "heart_disease_cleveland/019c7513-909c-707d-a9da-9852f346a015",
    "heart_disease_hungary/019c751a-ac4e-7446-83f1-6d3cd7333a80",
    "heart_disease_switzerland/019c751d-06af-7c95-aa04-957cd32c24f4",
    "heart_disease_va_long_beach/019c7521-3300-74cc-b795-e1b028bbd79f",
    "heloc/019d4456-12af-780b-8642-bbd1f65b0e77",
    "hepatitis_survival_prediction/019c7530-a96b-7845-a664-4c89f661b7c5",
    "hiva_agnostic/019d4457-cc30-71c7-bbef-6c46c871d114",
    "homesite_quote_conversion/019c718e-4b43-78a3-bc0f-1de37f8a9417",
    "horse_colic_survival/019c7554-7410-718c-b0ff-1d659f13c06b",
    "hr_analytics/019d445c-d0d0-7e98-8a96-979918c31004",
    "in_vehicle_coupon_recommendation/019d445e-db47-7d1a-9c30-e40eb064ba5e",
    "ionosphere/019d47b8-ec99-7c1c-91f5-20e67c05362e",
    "japanese_credit_screening/019cb04d-fef7-70be-949e-03181003021f",
    "kddcup09_appetency/019d47c4-d811-73ae-9229-83595e2522b2",
    "labor_negotiations/019d47c9-11b8-792d-bdb3-382fd970bf33",
    "leukemia_allaml/019d47cb-654d-786e-b923-45941a680429",
    "ljubljana_primary_tumor/019c80b5-f3a2-7254-890f-c728e070cfed",
    "lung_cancer/019d47ce-150a-70b9-85b8-1bb85244655d",
    "lymphography/019c75c5-725c-708c-9794-eccc46c0bf81",
    "marketing_campaign/019d47e1-1c2d-798b-ac11-f72a9d7f5f2f",
    "mechanisms_of_action/019d47e2-1aff-7ffb-9b99-61c8a74fc1a2",
    "miami_housing/019d47e2-cfb1-7f2c-ac5c-edb3c90067ac",
    "mic/019d47e3-6739-7346-9627-18b9b81cebca",
    "naticusdroid_android_permissions_dataset/019d47e5-2e69-7914-812d-ee84eeab35c9",
    "nci_ovarian_cancer/019d47ea-3ee8-71e4-850a-ace45ee7456f",
    "nci_pancreatic_cancer/019d47ed-572e-7228-84b6-b3d9a0cc35b7",
    "nci_prostate_cancer/019d47f4-eec6-7dec-882d-5e08502a1236",
    "online_shoppers_purchasing_intention_dataset/019d47f6-d29c-7895-a6c6-1f1bef149ee9",
    "otto_group_product_classification_challenge/019c10cf-1388-7c7b-a68b-5ebb4df6f33a",
    "polish_companies_bankruptcy/019d47fa-1094-7725-bf7b-e68cf54d9358",
    "porto_seguro/019c0ecf-ccbf-7572-9de8-626145c54342",
    "predict_students_dropout_and_academic_success/019d47fb-2264-7890-86c9-0c4d9a7e032a",
    "prostate_cancer/019d47fe-2699-7e75-b3cc-c11f60f8aef8",
    "pva_revenue_prediction_kddcup98/019c101e-100c-7def-aa83-b7a6b15eb4e3",
    "qsar_biodeg/019d4803-a135-7879-8007-9e3ffc50059b",
    "qsar_oral_toxicity/019d4801-956d-73e5-9f5a-3fb681555cc4",
    "qsar_tid_11/019d4801-f46b-70f0-ac4f-a064e923ac10",
    "santander_customer_transaction_prediction/019c1102-f88b-7cdd-9f59-2509e8bef0a7",
    "sdss_17/019d4806-d7f4-7238-9dab-3961ba5d3927",
    "smoking_lung_cancer/019d480b-2ab4-7865-81c4-fdbf8b74d482",
    "soybean_large/019d480e-44db-74ca-a302-744fa35cd88e",
    "srbct_prediction/019d4811-ccd1-7753-8352-908101430ed0",
    "superconductivity/019d4807-5d5b-7b50-bbaf-049008bcbad1",
    "taiwanese_bankruptcy_prediction/019d4813-8e54-7165-bf85-1255476188e1",
    "thyroid_discordant/019c7a7b-48ff-7471-8c09-3a07b8434a16",
    "wine_quality/019d4816-2a31-7d60-a95f-2f6167c8e008",
]


def user_task_id(task_name: str) -> int:
    """The id TabArena gives the ``UserTask`` of a Data Foundry container (``<dataset>/<uuid>``).

    Same formula as ``tabarena.benchmark.task.UserTask.task_id``; results store it as the task id.
    """
    return int(hashlib.sha256(task_name.encode("utf-8")).hexdigest(), 16) % 10**10


#: Task id of every known container -> dataset name.
TASK_ID_TO_DATASET: dict[int, str] = {
    user_task_id(path): path.split("/")[0] for path in [*EARLIER_DATA_FOUNDRY_TASKS, *BENCHMARK_DATA_FOUNDRY_TASKS]
}

#: Hash part of TabArena's task slug (``UserTask.slug``, ``<dataset>-<first 12 hex chars of the
#: task name's SHA-256>``) of every known container -> dataset name. The slug is the task name in
#: the results; its dataset part is cut to 40 characters, so the hash identifies the dataset.
TASK_SLUG_HASH_TO_DATASET: dict[str, str] = {
    hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]: path.split("/")[0]
    for path in [*EARLIER_DATA_FOUNDRY_TASKS, *BENCHMARK_DATA_FOUNDRY_TASKS]
}


_TASK_ID = re.compile(r"(?:Task-)?(\d+)(?:\.0)?")
_TASK_SLUG = re.compile(r".+-([0-9a-f]{12})")


def task_names_from_ids(task_id_strs) -> dict[int, str]:
    """Task id -> dataset name from TabArena ``task_id_str``s (``UserTask|<id>|<dataset>/<uuid>``),
    e.g. the ``data_foundry_task_id`` column of the stability / validity results."""
    names = {}
    for value in task_id_strs:
        parts = str(value).split("|")
        if len(parts) > 2 and parts[0] == "UserTask" and parts[1].isdigit():
            names[int(parts[1])] = parts[2].split("/")[0]
    return names


def dataset_name(task, known: Mapping[int, str] | None = None) -> str:
    """Dataset name of whatever a result file uses to identify the task.

    Accepts a dataset name (``anneal``), a container path (``anneal/<uuid>``), a TabArena
    ``task_id_str`` (``UserTask|<id>|anneal/<uuid>``), a task name as TabArena writes it into the
    results (the slug ``anneal-3f2a9c41d0b7``, or ``Task-1386903908`` in the fork's TabArena) or a
    bare task id (``1386903908``). Ids are looked up in ``known`` and in :data:`TASK_ID_TO_DATASET`,
    slugs in :data:`TASK_SLUG_HASH_TO_DATASET`; an unknown id or slug is returned unchanged, as a
    string.
    """
    text = str(task).strip()
    if text.startswith("UserTask|"):
        parts = text.split("|")
        text = parts[2] if len(parts) > 2 else parts[1]
    if "/" in text:
        return text.split("/")[0]
    match = _TASK_ID.fullmatch(text)
    if match:
        task_id = int(match.group(1))
        return (known or {}).get(task_id) or TASK_ID_TO_DATASET.get(task_id, text)
    slug = _TASK_SLUG.fullmatch(text)
    if slug:
        return TASK_SLUG_HASH_TO_DATASET.get(slug.group(1), text)
    return text


#: Hugging Face dataset repository holding the suite's containers, in the warehouse layout
#: ``<dataset>/<container uuid>/`` (as ``TabArena/BeyondArena`` holds BeyondArena's).
HF_REPO_ID = "SelectArena/SelectArena"
#: Commit of :data:`HF_REPO_ID` the containers are read from. Pinned so the files behind a container
#: path cannot change; update it whenever the repository gets new containers. ``None`` reads ``main``.
HF_REVISION: str | None = None


def make_collection(warehouse_dir: str | Path | None = None) -> DatasetCollection:
    """The suite as a Data Foundry collection.

    The containers are downloaded from :data:`HF_REPO_ID` into the Data Foundry cache
    (``$DATA_FOUNDRY_CACHE``, set by the ``data_foundry`` entry of ``CacheConfig``). With
    ``warehouse_dir`` they are read from a local warehouse instead, at
    ``<warehouse_dir>/<dataset>/<container uuid>``.
    """
    from data_foundry.collections import DatasetCollection, HuggingFaceSource, LocalWarehouseSource

    if warehouse_dir is None:
        source = HuggingFaceSource(repo_id=HF_REPO_ID, revision=HF_REVISION)
    else:
        source = LocalWarehouseSource(base_dir=Path(warehouse_dir))
    return DatasetCollection.from_relative_paths(
        name=COLLECTION_NAME,
        description=COLLECTION_DESCRIPTION,
        relative_paths=list(BENCHMARK_DATA_FOUNDRY_TASKS),
        source=source,
    )


def prepare_datasets(
    *,
    warehouse_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    openml_cache: str | Path | None = None,
) -> Path:
    """Convert the suite into TabArena ``UserTask``s and write its metadata CSV; return the CSV path.

    The containers come from Hugging Face (downloaded into ``cache_dir``, by default the Data
    Foundry cache) or, with ``warehouse_dir``, from a local warehouse (see :func:`make_collection`).
    """
    from tabarena.benchmark.task.data_foundry.metadata_cache import prepare_collection_for_tabarena

    return prepare_collection_for_tabarena(
        collection=make_collection(warehouse_dir),
        benchmark_suite_name=COLLECTION_NAME,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        openml_cache=openml_cache,
    )


def make_task_metadata_collection(
    *, warehouse_dir: str | Path | None = None, cache_dir: str | Path | None = None
) -> TaskMetadataCollection:
    """The suite's ``TaskMetadataCollection``.

    Reads the metadata CSV written by :func:`prepare_datasets` (from the OpenML cache the active
    ``CacheConfig`` points at). If it is missing, the suite is converted first, from Hugging Face
    or from ``warehouse_dir`` (see :func:`make_collection`).
    """
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.sources.data_foundry import DataFoundryTaskMetadataSource

    source = DataFoundryTaskMetadataSource(
        collection_factory=partial(make_collection, warehouse_dir),
        name=COLLECTION_NAME,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return TaskMetadataCollection.from_source(source)
