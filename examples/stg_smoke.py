"""Held-out smoke evaluation of tuned STG on three bundled real datasets.

    python examples/stg_smoke.py --output private/stg_converged_smoke.json

Uses convergence-aware STG tuning, a fixed 75/25 outer split, ceil(d/3) features and
fixed LightGBM predictors. STG receives ONLY outer training rows. All features
and seeded random subsets are controls. This is a smoke evaluation, not the
published SelectArena dataset suite or a leaderboard result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from selectarena.methods.stg import STGFeatureSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("private/stg_converged_smoke.json"))
    parser.add_argument("--time-limit", type=float, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dataset", choices=["all", "breast_cancer", "wine", "diabetes"], default="all")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-lambda-trials", type=int)
    parser.add_argument("--initial-lambdas", nargs="+", type=float)
    args = parser.parse_args()
    results = []
    for loader, problem, metric in [
        (load_breast_cancer, "binary", "roc_auc"),
        (load_wine, "multiclass", "log_loss"),
        (load_diabetes, "regression", "rmse"),
    ]:
        if args.dataset != "all" and loader.__name__ != f"load_{args.dataset}":
            continue
        dataset = loader(as_frame=True)
        X, y = dataset.data, dataset.target
        regression = problem == "regression"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=args.seed, stratify=None if regression else y
        )
        budget = math.ceil(X.shape[1] / 3)
        settings = {}
        if args.max_epochs is not None:
            settings["max_epochs"] = args.max_epochs
        if args.max_lambda_trials is not None:
            settings["max_lambda_trials"] = args.max_lambda_trials
        if args.initial_lambdas is not None:
            settings["lambdas"] = args.initial_lambdas
        selector = STGFeatureSelector(
            max_features=budget, random_state=args.seed, verbosity=0, device=args.device, **settings
        )
        start = time.monotonic()
        selected = selector.fit_transform(X=X_train, y=y_train, problem_type=problem, time_limit=args.time_limit)
        elapsed = time.monotonic() - start
        features = {
            "Random": np.random.default_rng(args.seed).choice(X.columns, size=budget, replace=False).tolist(),
            "All": list(X.columns),
        }
        valid = selector.budget_converged_ and not selector._feature_selection_random_features
        if valid:
            features["STG"] = list(selected.columns)
        scores = {}
        for name, columns in features.items():
            cls = LGBMRegressor if regression else LGBMClassifier
            model = cls(n_estimators=100, n_jobs=1, random_state=args.seed, verbosity=-1)
            model.fit(X_train[columns], y_train)
            if regression:
                score = math.sqrt(mean_squared_error(y_test, model.predict(X_test[columns])))
            else:
                probabilities = model.predict_proba(X_test[columns])
                score = (
                    roc_auc_score(y_test, probabilities[:, 1])
                    if problem == "binary"
                    else log_loss(y_test, probabilities)
                )
            scores[name] = float(score)
        row = {
            "dataset": loader.__name__,
            "problem_type": problem,
            "metric": metric,
            "seed": args.seed,
            "requested_device": args.device,
            "search_config": {
                "initial_lambdas": selector.lambdas,
                "architectures": selector.architectures,
                "learning_rates": selector.learning_rates,
                "max_epochs": selector.max_epochs,
                "max_lambda_trials": selector.max_lambda_trials,
                "min_epochs": selector.min_epochs,
                "gate_window": selector.gate_window,
                "gate_tolerance": selector.gate_tolerance,
                "budget_tolerance": selector.budget_tolerance,
            },
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "budget": budget,
            "feature_count": X.shape[1],
            "scores": scores,
            "selected_features": list(selected.columns) if valid else [],
            "evaluation_status": "converged_budget_match" if valid else "not_evaluated_no_converged_budget_match",
            "native_active_count": selector.native_count_,
            "budget_converged": selector.budget_converged_,
            "gate_diagnostics": selector.gate_diagnostics_,
            "gate_means": selector.gate_means_.astype(float).tolist() if valid else None,
            "selection_seconds": elapsed,
            "best_params": selector.best_params_,
            "best_epoch": selector.best_epoch_,
            "best_validation_loss": selector.best_validation_loss_
            if math.isfinite(selector.best_validation_loss_)
            else None,
            "trials": selector.search_results_,
            "random_fills": selector._feature_selection_random_features,
        }
        results.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        print(
            pd.DataFrame(
                [
                    {
                        "dataset": row["dataset"],
                        "metric": metric,
                        "status": row["evaluation_status"],
                        "native_active": row["native_active_count"],
                        **scores,
                    }
                ]
            ).to_string(index=False),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved tuning traces and held-out scores to {args.output}")


if __name__ == "__main__":
    main()
