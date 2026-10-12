"""The SelectArena dataset suite and how it becomes a TabArena task collection.

The suite is a list of Data Foundry containers (``<dataset>/<container uuid>``). They are
hosted on Hugging Face (:data:`HF_REPO_ID`) or read from a local Data Foundry warehouse, and
converted once into TabArena ``UserTask``s with :func:`prepare_datasets` (writes the task pickles
into the OpenML cache and a ``<COLLECTION_NAME>_tasks_metadata.csv`` next to them); afterwards
:func:`make_task_metadata_collection` reads that CSV without touching Data Foundry again.

Requires ``tabarena[data-foundry]`` for the conversion step.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_foundry.collections import DatasetCollection
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

COLLECTION_NAME = "feature_selection_benchmark"
COLLECTION_DESCRIPTION = "SelectArena: tabular datasets for benchmarking feature selection methods."
#: Hugging Face dataset repo hosting the containers (layout ``<dataset>/<uuid>/``).
HF_REPO_ID = "SelectArena/datasets"

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


def make_collection(warehouse_dir: str | Path | None = None, *, repo_id: str = HF_REPO_ID) -> DatasetCollection:
    """The suite as a Data Foundry collection.

    With ``warehouse_dir``, the containers are read from a local warehouse
    (``<warehouse_dir>/<dataset>/<container uuid>``); otherwise they are downloaded on demand from
    the Hugging Face dataset repo ``repo_id``, which mirrors that layout (see
    :mod:`selectarena.data_release`).
    """
    from data_foundry.collections import DatasetCollection, HuggingFaceSource, LocalWarehouseSource

    source = (
        LocalWarehouseSource(base_dir=Path(warehouse_dir).expanduser())
        if warehouse_dir is not None
        else HuggingFaceSource(repo_id=repo_id)
    )
    return DatasetCollection.from_relative_paths(
        name=COLLECTION_NAME,
        description=COLLECTION_DESCRIPTION,
        relative_paths=list(BENCHMARK_DATA_FOUNDRY_TASKS),
        source=source,
    )


def prepare_datasets(*, warehouse_dir: str | Path | None = None, openml_cache: str | Path | None = None) -> Path:
    """Convert the suite into TabArena ``UserTask``s and write its metadata CSV; return the CSV path.

    Reads the containers from ``warehouse_dir``, or downloads them from Hugging Face without it.
    """
    from tabarena.benchmark.task.data_foundry.metadata_cache import prepare_collection_for_tabarena

    return prepare_collection_for_tabarena(
        collection=make_collection(warehouse_dir),
        benchmark_suite_name=COLLECTION_NAME,
        openml_cache=openml_cache,
    )


def make_task_metadata_collection(
    *, warehouse_dir: str | Path | None = None, cache_dir: str | Path | None = None
) -> TaskMetadataCollection:
    """The suite's ``TaskMetadataCollection``.

    Reads the metadata CSV written by :func:`prepare_datasets` (from the OpenML cache the active
    ``CacheConfig`` points at). If it is missing, the suite is converted first, from
    ``warehouse_dir`` or, without it, from Hugging Face.
    """
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.sources.data_foundry import DataFoundryTaskMetadataSource

    source = DataFoundryTaskMetadataSource(
        collection_factory=partial(make_collection, warehouse_dir),
        name=COLLECTION_NAME,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return TaskMetadataCollection.from_source(source)
