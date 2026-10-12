"""Abstract base class for feature selection methods."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.data import LabelCleaner
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from autogluon.tabular.models.lgb.lgb_model import LGBModel

if TYPE_CHECKING:
    import pandas as pd
    from autogluon.core.models.abstract.abstract_model import AbstractModel


@dataclass
class ProxyModelConfig:
    """Configuration for the proxy model used in feature selection methods."""

    model_class: AbstractModel = LGBModel
    """Proxy Model class to use within feature selection methods."""
    model_hyperparameters: dict = field(default_factory=dict)
    """Hyperparameters to use when fitting the proxy model. Empty dict is the default."""
    use_bagged_model: bool = False
    """Whether to use a bagged model as the proxy model."""
    bagging_hyperparameters: dict = field(default_factory=dict)
    """Hyperparameters to set for the bagging model (such as refitting or not)."""
    verbosity: int = 0
    """Verbosity to use when fitting the proxy model."""
    problem_type: str | None = None
    """The problem type of the current datasets."""
    eval_metric: str | None = None
    """The evaluation metric used to evaluate the proxy model."""


class AbstractFeatureSelector(AbstractFeatureGenerator):
    """Abstract Feature Selector."""

    name: str
    """The name of the feature selector. This is used for logging and debugging purposes, and should be
    overridden by each feature selector."""
    feature_scoring_method: bool = False
    """If True, the method computes feature scores (higher is better) for features, instead
    of directly selecting features. The subclass only needs to implement logic that fill _feature_scores.
    If False, the method cannot compute feature scores, and directly fill _selected_features.
    """

    max_features: int
    """The maximum number of features to select."""

    _original_features: list[str]
    """The list of original features before fitting the feature selector."""

    _selected_features: list[dict[str, float | None]] | None
    """The finally-selected features, as an ordered list of dicts (selection order, strongest
    first). Each entry is {"feature": <name>, "value": <criterion value>}, where "value" follows
    a three-state convention:
        - float : the feature's value under the method's own criterion (the score for scoring
                  methods, the merit/CMI/relevance gain for greedy subset methods, ...).
        - NaN   : the feature was selected by the method, but the method exposes no per-feature
                  scalar (e.g. backward elimination: SBE, Consistency, MarkovBlanket).
        - None  : the feature was added by the random fallback (pure padding, no criterion).
    None (the attribute itself) if the selector is not fitted yet.
    """

    _feature_scores: dict[str, float] | None
    """Mapping from feature name to score (higher is more important).
    None if the feature selector is not fitted yet, or if the method does not compute feature scores.
    Note: this mapping may be incomplete, depending on the method and
    if the method finished evaluating all features within the time limit.
    """

    _feature_selection_fit_time: int | None
    """Time it took to perform the feature selection"""

    _feature_selection_time_limit: int | None
    """Time limit for the feature selection method."""

    _feature_selection_random_features: list[str]
    """Names of the features in the final selection that were chosen at random by the
    fallback (forward search: the randomly added fills; backward search: the randomly
    kept subset). Empty if the fallback was never triggered."""

    _rng: np.random.Generator
    """Random number generator for fallback feature selection."""

    _outer_random_state: int | None
    """Per-split seed set by the benchmark wrapper before fitting (see
    ``selectarena.benchmark.wrapper``). When set, it overrides ``random_state`` so the random
    fallback and internal splits differ across outer folds and repeats."""

    problem_type: str | None
    """The problem type of the current dataset"""

    def __init__(
        self,
        max_features: int | float | Callable[[int], int],
        *,
        proxy_mode_config: ProxyModelConfig | None = None,
        raise_on_useless_feature_selection: bool = True,
        skip_random_fallback: bool = False,
        encoding_permutation_seed: int | None = None,
        **kwargs,
    ):
        """Init from super class with additional feature selection specific parameters.

        Parameters
        ----------
        max_features: int, float, or Callable[[int], int]
            The maximum number of features to select.
            If int, we assume a fixed number of features to select.
            If float, we assume a fraction of features to select (0 < max_features < 1).
                We always round up.
            If callable, it receives the total number of input features as its single argument
                and must return the desired feature count as an int. Resolved at fit time.
        proxy_mode_config:
            Configuration of the proxy model to use inside the feature selection method.
        raise_on_useless_feature_selection:
            If True, the method raises an error when the input data contains less than max_features.
        encoding_permutation_seed:
            Ordinal-encoding ablation. If None (default), categorical columns are ordinal-encoded
            in the usual sorted category order. If set to an int, the category -> integer mapping
            of every ordinal-encoded column is randomly permuted with a dedicated RNG seeded by this
            value. This RNG is independent of ``self.random_state`` (which seeds the random fallback
            and internal train/val splits), so variance across seeds isolates the effect of the
            arbitrary ordinal assignment. See ``_encode_ordinal``.

        """
        super().__init__(**kwargs)

        self.max_features_input = max_features
        self.proxy_mode_config = proxy_mode_config
        self.raise_on_useless_feature_selection = raise_on_useless_feature_selection

        self._selected_features = None
        self._feature_scores = None
        self._outer_random_state = None
        self._feature_selection_fit_time = None
        self._feature_selection_time_limit = None
        self._feature_selection_random_features = []
        self.skip_random_fallback = skip_random_fallback
        self.encoding_permutation_seed = encoding_permutation_seed

    def _fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        time_limit: int | None = None,
        eval_metric: str | None = None,
        problem_type: str | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Fit and transform for feature selection methods."""
        if self._outer_random_state:
            self.random_state = self._outer_random_state

        self._original_features = list(X.columns)
        old_type_family_groups_special = {}
        if self.feature_metadata_in.type_group_map_special is not None:
            old_type_family_groups_special = deepcopy(self.feature_metadata_in.type_group_map_special)

        # Init random generator
        self._rng = np.random.default_rng(self.random_state)
        self.problem_type = problem_type
        self._feature_selection_time_limit = time_limit

        self._resolve_proxy_config(eval_metric=eval_metric, problem_type=problem_type)
        self._resolve_max_features()

        # Sanity check for feature selection setup
        if len(self._original_features) <= self.max_features:
            if self.raise_on_useless_feature_selection:
                raise ValueError(
                    "The number of features in the input data is less than or equal to max_features. "
                    "Feature selection has no affect on the pipeline in this case. "
                    "Set raise_on_useless_feature_selection to False to disable this check."
                )
            return X, old_type_family_groups_special

        # Call feature selection method
        feature_fit_kwargs = dict(X=X, y=y, time_limit=time_limit)
        start_time = time.monotonic()
        if self.feature_scoring_method:
            self._feature_scores = self._fit_feature_scoring(**feature_fit_kwargs)
            assert isinstance(self._feature_scores, dict), (
                "The feature scores must be a dictionary mapping from feature name to score."
            )
            self._selected_features = self.selected_features_from_feature_scores()
        else:
            self._selected_features = self._fit_feature_selection(**feature_fit_kwargs)
            assert isinstance(self._selected_features, list), (
                "The selected features must be a list of {feature, value} dicts."
            )
            if len(self._selected_features) != self.max_features:
                self._log(
                    30,
                    f"Warning: Selected {len(self._selected_features)} features with a budget of {self.max_features}. "
                    f"Fallback to random selection.",
                )
                self._selected_features = self.fallback_feature_selection(selected_features=self._selected_features)
        assert isinstance(self._selected_features, list), (
            "The selected features must be a list of {feature, value} dicts."
        )
        if self.skip_random_fallback:
            assert len(self._selected_features) >= 1
        else:
            assert len(self._selected_features) == self.max_features, (
                f"Expected {self.max_features} selected features, got {len(self._selected_features)}."
            )

        self._feature_selection_fit_time = time.monotonic() - start_time

        # Transform (aka select features)
        selected_names = [entry["feature"] for entry in self._selected_features]
        X_out = self._transform(X=X)
        assert list(X_out) == selected_names, "The output features must be the same as the selected features."

        # Update the type family groups in the output feature metadata.
        type_family_groups_special: dict[str, list[str]] = {
            type_group_name: [f for f in feature_names if f in selected_names]
            for type_group_name, feature_names in old_type_family_groups_special.items()
        }

        return X_out, type_family_groups_special

    def _resolve_proxy_config(self, *, eval_metric: str | None, problem_type: str | None) -> None:
        """Synchronize proxy model config with the given eval_metric/problem_type."""
        if self.proxy_mode_config is None:
            return
        assert isinstance(self.proxy_mode_config, ProxyModelConfig)
        if self.proxy_mode_config.problem_type is not None:
            problem_type = self.proxy_mode_config.problem_type
        if self.proxy_mode_config.eval_metric is not None:
            eval_metric = self.proxy_mode_config.eval_metric
        self.proxy_mode_config.problem_type = problem_type
        self.proxy_mode_config.eval_metric = eval_metric

    def _resolve_max_features(self) -> None:
        """Resolve max_features from int, fraction, or callable."""
        if callable(self.max_features_input):
            self.max_features = self.max_features_input(len(self._original_features))
            self._log(20, f"Resolved max_features to {self.max_features} via callable.")
        elif isinstance(self.max_features_input, float):
            if not (0 < self.max_features_input < 1):
                raise ValueError("If max_features is a float, it must be in the range (0, 1).")
            self.max_features = math.ceil(len(self._original_features) * self.max_features_input)
            self._log(20, f"Resolved max_features to {self.max_features} based on the fraction provided.")
        else:
            self.max_features = self.max_features_input

    def _fit_feature_selection(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[dict[str, float | None]]:
        """Code that fills self._selected_features. Used if feature_scoring_method is False.

        Must return an ordered list (selection order, strongest first) of
        {"feature": <name>, "value": <criterion value>} dicts. "value" is a float with the
        feature's value under the method's own criterion, or NaN (``float("nan")``) when the
        method selects the feature but exposes no per-feature scalar. Do not use None here;
        None is reserved for random-fallback fills added by the abstract layer.
        """
        raise NotImplementedError

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        """Code that fills self._feature_scores. Used if feature_scoring_method is True."""
        raise NotImplementedError

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data according to the selected features."""
        if self._selected_features is None:
            raise ValueError("Feature selector has not been fitted yet. Please call fit_transform first.")

        selected_names = [entry["feature"] for entry in self._selected_features]

        # Check if all of selected features is in X
        missing_features = [feature for feature in selected_names if feature not in X.columns]
        if missing_features:
            raise ValueError(f"Some of the selected features are not in the input data: {missing_features}")

        return X[selected_names]

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        """By default, all our feature selector want to get all data as input.

        Note: as we assume that we call the feature selector as a post generator,
        all data should be in a proper format for the feature selector.
        TODO: if not, determine how to adapt the code that the feature selector performs
            its own preprocessing.
        """
        return dict()

    # TODO: investigate how to improve randomness of this fallback method
    #   Note: by default, the column order of input features is likely always the same.
    #   Thus, the randomly selected features are likely always the same across different runs,
    #   which is not ideal for a fallback method. So likely, we want to somehow adjust the random state
    #   based on some property of the input data, or pass a different random state per run.
    def fallback_feature_selection(
        self, *, selected_features: list[dict[str, float | None]] | None = None
    ) -> list[dict[str, float | None]]:
        """Randomly select features as a fallback with support for partial fallback.

        Parameters
        ----------
        selected_features: list[dict] | None
            The already-selected features (as {"feature", "value"} dicts) that should be included
            in the output. If None, no features are guaranteed to be included in the output.

        Returns
        -------
        list[dict]
            The selection as {"feature", "value"} dicts. Randomly chosen top-ups carry a
            "value" of None. ``_feature_selection_random_features`` is set to their names.
        """
        if self.skip_random_fallback and selected_features:
            return selected_features
        if selected_features is None:
            selected_features = []
        selected_names = [entry["feature"] for entry in selected_features]

        if len(selected_features) >= self.max_features:  # backward search
            kept = [str(f) for f in self._rng.choice(selected_names, size=self.max_features, replace=False)]
            self._feature_selection_random_features = kept
            return [{"feature": f, "value": None} for f in kept]

        # forward search
        to_select_from_features = [feature for feature in self._original_features if feature not in selected_names]
        num_feature_to_select = self.max_features - len(selected_features)
        rand_features = [
            str(f) for f in self._rng.choice(to_select_from_features, size=num_feature_to_select, replace=False)
        ]
        self._feature_selection_random_features = rand_features
        return selected_features + [{"feature": f, "value": None} for f in rand_features]

    def selected_features_from_feature_scores(self) -> list[dict[str, float | None]]:
        """Convert feature scores to the selected-feature structure.

        Returns an ordered list of {"feature": name, "value": score} dicts (highest score first).
        If not enough features have scores to fill the budget, the remainder is filled by the
        random fallback; those top-up features carry a "value" of None.
        """
        feature_scores = self._feature_scores
        if feature_scores is None:
            raise ValueError("Feature scores are not computed yet. Please call _fit_feature_scoring first.")

        # Sort features by score (higher is better) and select the top max_features
        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)
        selected_names = sorted_features[: self.max_features]
        selection = [{"feature": f, "value": float(feature_scores[f])} for f in selected_names]

        # Fallback for missing feature scores
        if len(selection) < self.max_features:
            self._log(
                30,
                f"Warning: Not enough feature scores computed to select {self.max_features} features. "
                f"Selected {len(selection)} features based on available scores, "
                f"and randomly selected the rest.",
            )

            selection = self.fallback_feature_selection(selected_features=selection)

        return selection

    # TODO: make this a standalone static method re-usable across TabArena/AutoGluon
    #   - Similar to code in autogluon.core.utils.feature_selection.FeatureSelector
    #     we can add logging, stability, more...
    #   - Add support for cross-validation (instead of holdout?)
    #   - Figure out how to handle TimeLimitExceeded raise in this code or the code that calls this.
    #       -> would need to skip step.
    def evaluate_proxy_model(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
        n_train_subsample: int | None = 10_000,
        n_val_fraction: float = 0.33,
    ) -> float | None:
        """Evaluate the proxy model on the given data."""
        from autogluon.core.models import BaggedEnsembleModel
        from autogluon.core.utils.exceptions import NoValidFeatures
        from autogluon.core.utils.utils import generate_train_test_split
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator

        if self.proxy_mode_config is None:
            raise ValueError(
                "Proxy mode is not configured but the feature selection method needs a proxy model. "
                "Pass a ProxyModelConfig to the feature selection method!"
            )
        assert self.proxy_mode_config.eval_metric is not None, "When ProxyModel is needed, eval_metric must be set."
        assert self.proxy_mode_config.problem_type is not None, "When ProxyModel is needed, problem_type must be set."
        self._log(20, f"\tFitting Proxy Model (remaining time_limit: {time_limit})")

        # Init Proxy model
        problem_type = self.proxy_mode_config.problem_type
        eval_metric = self.proxy_mode_config.eval_metric
        model_class = self.proxy_mode_config.model_class
        use_bagged_model = self.proxy_mode_config.use_bagged_model
        hps = self.proxy_mode_config.model_hyperparameters
        bagging_hps = self.proxy_mode_config.bagging_hyperparameters
        verbosity = self.proxy_mode_config.verbosity
        model_kwargs = dict(
            name=f"ProxyModel_{model_class.ag_name}",
            problem_type=problem_type,
            eval_metric=eval_metric,
            hyperparameters=hps,
        )

        # Validation splits + Subsampling
        X, y = X.copy(), y.copy()  # avoid modifying the input data
        X_train, X_val, y_train, y_val = generate_train_test_split(
            X=X, y=y, test_size=n_val_fraction, random_state=self.random_state, problem_type=problem_type
        )
        del X, y  # free up memory
        if (n_train_subsample is not None) and (len(X_train) > n_train_subsample):
            self._log(
                20,
                f"\tNumber of training samples {len(X_train)} is greater than {n_train_subsample}. "
                f"Using {n_train_subsample} samples as training data.",
            )
            drop_ratio = 1.0 - n_train_subsample / len(X_train)
            X_train, _, y_train, _ = generate_train_test_split(
                X=X_train, y=y_train, problem_type=problem_type, random_state=self.random_state, test_size=drop_ratio
            )

        # Preprocessing
        feature_generator, label_cleaner = (
            AutoMLPipelineFeatureGenerator(verbosity=verbosity),
            LabelCleaner.construct(problem_type=problem_type, y=y_train, verbose=verbosity > 1),
        )
        X_train, y_train = (
            feature_generator.fit_transform(X_train),
            label_cleaner.transform(y_train),
        )
        X_val, y_val = feature_generator.transform(X_val), label_cleaner.transform(y_val)

        # Run proxy model
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_kwargs["path"] = tmp_dir
                if use_bagged_model:
                    model = BaggedEnsembleModel(
                        model_class(**model_kwargs),
                        hyperparameters=bagging_hps,
                    )
                else:
                    model = model_class(**model_kwargs)
                model.rename("FeatureSelector_" + model.name)
                model.fit(
                    X=X_train,
                    y=y_train,
                    time_limit=time_limit,
                    num_classes=label_cleaner.num_classes,
                    label_cleaner=label_cleaner,
                )
                score = model.score(X=X_val, y=y_val)
        except NoValidFeatures:
            self._log(
                30,
                "Proxy model received no valid features (all dropped as constant after split). Returning None.",
            )
            return None
        # Ensure Python dtype
        return float(score)

    def _seeded_feature_order(self) -> list[str]:
        """Original features in a per-split-seeded random order (``self._rng`` is seeded from the
        split index). For methods that iterate and may truncate on a time limit, so a timeout
        leaves a random scored subset instead of always the first columns."""
        idx = self._rng.permutation(len(self._original_features))
        return [self._original_features[i] for i in idx]

    def _timed_out(self, time_limit, start_time) -> bool:
        if time_limit is None:
            return False
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit:
            self._log(
                30,
                f"Warning: FeatureSelection Method has no time left to train... "
                f"\t(Time Elapsed = {elapsed:.1f}s, Time Limit = {time_limit:.1f}s)",
            )
            return True
        return False

    @staticmethod
    def _impute(
        X: pd.DataFrame,
        numeric_strategy: str = "mean",
        categorical_strategy: str = "most_frequent",
    ) -> pd.DataFrame:
        """Impute missing values separately for numeric and categorical features.

        Parameters
        ----------
        X: pd.DataFrame
            Input data with potential missing values.
        numeric_strategy: str
            Strategy for numeric columns (mean, median, most_frequent, constant).
        categorical_strategy: str
            Strategy for categorical columns (most_frequent, constant).

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values imputed.
        """
        from sklearn.impute import SimpleImputer

        X_imputed = X.copy()
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(num_cols):
            X_imputed[num_cols] = SimpleImputer(strategy=numeric_strategy).fit_transform(X[num_cols])
        if len(cat_cols):
            X_imputed[cat_cols] = SimpleImputer(strategy=categorical_strategy).fit_transform(X[cat_cols])

        return X_imputed

    @staticmethod
    def _discretize(X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import KBinsDiscretizer

        X_disc = X.copy()
        num_cols = X.select_dtypes(include="number").columns
        max_bins = int(np.log2(len(X))) + 1  # sturges' rule to scale with size

        for col in num_cols:
            n_bins = min(X[col].nunique(), max_bins)
            if n_bins < 2:
                continue  # constant feature, skip
            X_disc[col] = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf"
            ).fit_transform(X[[col]])
        return X_disc

    @staticmethod
    def _encode_ordinal(
        X: pd.DataFrame,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        permutation_seed: int | None = None,
    ) -> pd.DataFrame:
        """Ordinal-encode object/category columns.

        With ``permutation_seed=None`` (default) each column's categories are encoded in the
        usual sorted order. With an int ``permutation_seed`` the (arbitrary) category -> integer
        mapping of every categorical column is randomly permuted. The permutation is driven by a
        dedicated ``np.random.default_rng(permutation_seed)`` so it does not touch the selector's
        ``self.random_state`` (fallback / internal splits): this is the ordinal-encoding ablation
        knob, and variance across seeds measures sensitivity to the arbitrary ordinal assignment.
        """
        from sklearn.preprocessing import OrdinalEncoder

        X_enc = X.copy()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            if permutation_seed is None:
                encoder = OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=unknown_value)
            else:
                # Discover the per-column categories exactly as the default encoder would (sorted),
                # then shuffle each column's order; that shuffled order defines the integer codes.
                base_categories = OrdinalEncoder().fit(X[cat_cols]).categories_
                rng = np.random.default_rng(permutation_seed)
                permuted_categories = [rng.permutation(cats) for cats in base_categories]
                encoder = OrdinalEncoder(
                    categories=permuted_categories,
                    handle_unknown=handle_unknown,
                    unknown_value=unknown_value,
                )
            X_enc[cat_cols] = encoder.fit_transform(X[cat_cols])
        return X_enc

    @staticmethod
    def _encode_target(y: pd.Series) -> pd.Series:
        from sklearn.preprocessing import LabelEncoder

        return pd.Series(LabelEncoder().fit_transform(y), index=y.index)

    @staticmethod
    def _scale(X: pd.DataFrame) -> pd.DataFrame:
        """Standardize features to zero mean and unit variance."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(X)
        return pd.DataFrame(scaled_array, columns=X.columns, index=X.index)

    def _preprocess(
        self,
        X: pd.DataFrame,
        *,
        y: pd.Series | None = None,
        impute: bool = False,
        discretize: bool = False,
        encode_ordinal: bool = False,
        encode_target: bool = False,
        scale: bool = False,
        impute_kwargs: dict | None = None,
        discretize_kwargs: dict | None = None,
        encode_ordinal_kwargs: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Convenience method bundling common preprocessing."""
        if impute:
            X = self._impute(X, **(impute_kwargs or {}))
        if discretize:
            X = self._discretize(X, **(discretize_kwargs or {}))
        if encode_ordinal:
            # Default the permutation seed to the selector-level ablation knob; an explicit
            # permutation_seed in encode_ordinal_kwargs still wins if a caller sets one.
            eo_kwargs = {"permutation_seed": self.encoding_permutation_seed, **(encode_ordinal_kwargs or {})}
            X = self._encode_ordinal(X, **eo_kwargs)
        if encode_target:
            if y is None:
                raise ValueError("encode_target=True requires y to be provided.")
            y = self._encode_target(y)
        if scale:
            X = self._scale(X)
        return X, y


class AbstractITFeatureSelector(AbstractFeatureSelector):
    """Base class for information-theory-based feature selectors.

    Provides shared entropy, mutual information, and symmetrical uncertainty
    helpers used by MI, IG, JMI, CMIM, DISR, and CFS.
    """

    def _mutual_information_dispatched(self, x: pd.Series, y: pd.Series, base: float = 2) -> float:
        """
        Helper for calculating MI for continuous target (requires density function estimation for entropy.
        Used in GainRatio and mRMR.
        """
        if self.problem_type == "regression":
            from sklearn.feature_selection import mutual_info_regression

            mi_nat_log = mutual_info_regression(
                x.values.reshape(-1, 1),
                y.values,
                discrete_features=True,
                random_state=self.random_state,
            )[0]
            return mi_nat_log / np.log(base)
        return self._mutual_information(x, y, base=base)

    @staticmethod
    def _entropy(*xs: pd.Series, base: float = 2) -> float:
        """H(X1, X2, ..., Xk) — joint discrete entropy."""
        if len(xs) == 1:
            probs = xs[0].value_counts(normalize=True, dropna=False).values
        else:
            joint = pd.concat(xs, axis=1)
            probs = joint.value_counts(normalize=True, dropna=False).values
        if len(probs) <= 1:
            return 0.0
        return -np.sum(probs * np.log(probs)) / np.log(base)

    @classmethod
    def _mutual_information(cls, x: pd.Series, y: pd.Series, base: float = 2) -> float:
        """I(X; Y) = H(X) + H(Y) - H(X, Y)."""
        return cls._entropy(x, base=base) + cls._entropy(y, base=base) - cls._entropy(x, y, base=base)

    @classmethod
    def _conditional_mutual_information(cls, x, y, z, base: float = 2) -> float:
        """I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)."""
        return (
            cls._entropy(x, z, base=base)
            + cls._entropy(y, z, base=base)
            - cls._entropy(x, y, z, base=base)
            - cls._entropy(z, base=base)
        )

    @classmethod
    def _joint_mutual_information(self, x1, x2, y, base: float = 2):
        """I((X1, X2); Y) = H(X1, X2) + H(Y) - H(X1, X2, Y)"""
        return self._entropy(x1, x2, base=base) + self._entropy(y, base=base) - self._entropy(x1, x2, y, base=base)

    @classmethod
    def _symmetrical_uncertainty(cls, x, y, base: float = 2) -> float:
        """SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))."""
        hx = cls._entropy(x, base=base)
        hy = cls._entropy(y, base=base)
        if hx + hy == 0:
            return 0.0
        ixy = hx + hy - cls._entropy(x, y, base=base)
        return 2.0 * ixy / (hx + hy)

    @classmethod
    def _symmetrical_relevance(cls, x1, x2, y, base: float = 2) -> float:
        """Normalized joint MI: I((X1, X2); Y) / H(X1, X2, Y). Used by DISR.

        Returns 0 if joint entropy is 0 (degenerate case).
        """
        joint_entropy = cls._entropy(x1, x2, y, base=base)
        if joint_entropy == 0:
            return 0.0
        joint_mi = cls._joint_mutual_information(x1, x2, y, base=base)
        return joint_mi / joint_entropy
