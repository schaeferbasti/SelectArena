"""CD-PFN feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from selectarena.methods.abstract import AbstractFeatureSelector


class CDPFNFeatureSelector(AbstractFeatureSelector):
    """CD-PFN Feature Selection.

    Scores each feature by CD-PFN's posterior probability that the target depends on it given the other features,
    from a pretrained prior-data fitted network. Multiclass targets are scored by the mean of the one-vs-rest
    probabilities; rare classes, features with extreme values, and large inputs are handled inside the package (see
    ``help(varselpfn.CDPFN)``). A one-shot call that cannot be
    interrupted: about a minute at 10,000 rows and 100 features on a laptop CPU, longer for thousands of features.

    Reference: Melikechi, Omar. "Prior-data fitted networks for variable selection." arXiv:2611.05966 (2026).
    Implementation: the varselpfn package (https://github.com/omelikechi/varselpfn), MIT licence.
    """

    name = "CDPFNFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: int | None = None,
    ) -> dict[str, float]:
        from varselpfn import CDPFN

        # categories as ordinal codes (so the encoding-permutation ablation applies); missing values are filled
        # inside the package
        X_pre, _ = self._preprocess(X, encode_ordinal=True)
        model = CDPFN(response=self.problem_type, seed=int(self.random_state or 0))
        pips = model.predict(X_pre.reset_index(drop=True), pd.Series(np.asarray(y)))
        return dict(zip(X_pre.columns, np.nan_to_num(pips.to_numpy(dtype=float), nan=0.0).tolist()))
