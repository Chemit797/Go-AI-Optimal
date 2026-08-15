"""Exact matched-control construction for diagnostics and training priors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import MATCH_CONTROL_FIELDS, control_mask


@dataclass
class ControlMatch:
    predictions: pd.DataFrame
    has_exact_match: pd.Series


def _keys(metadata: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(metadata.loc[:, MATCH_CONTROL_FIELDS].astype(str))


def exact_control_predictions(
    metadata: pd.DataFrame,
    y_log2: pd.DataFrame,
    target_ids: pd.Index | list[str],
    control_pool_ids: pd.Index | list[str] | None = None,
) -> ControlMatch:
    """Return per-protein means of controls sharing all documented match fields."""
    targets = pd.Index(target_ids)
    pool = metadata.index if control_pool_ids is None else pd.Index(control_pool_ids)
    eligible_controls = pool[control_mask(metadata.loc[pool]).to_numpy()]
    if len(eligible_controls) == 0:
        prediction = pd.DataFrame(index=targets, columns=y_log2.columns, dtype=float)
        return ControlMatch(prediction, pd.Series(False, index=targets, name="has_exact_match"))

    controls = metadata.loc[eligible_controls]
    control_means = y_log2.loc[eligible_controls].groupby(_keys(controls), sort=False).mean()
    target_keys = _keys(metadata.loc[targets])
    prediction = control_means.reindex(target_keys)
    prediction.index = targets
    has_match = pd.Series(target_keys.isin(control_means.index), index=targets, name="has_exact_match")
    return ControlMatch(prediction, has_match)
