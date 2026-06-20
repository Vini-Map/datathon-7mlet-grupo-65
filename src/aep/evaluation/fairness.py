"""Exposure-fairness analysis across synthetic segments (Stage 4).

For the frozen serving policy we measure, per synthetic segment, how offer
exposure is distributed and what expected conversion the policy delivers. Large
gaps flag that the policy concentrates value (or a product) on some segments —
important context for "when not to use" and for the suitability narrative.

`segment` is a synthetic construct (age band x white-collar), not a protected
attribute; this is an operational exposure check, not a legal fairness audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aep.bandits.serving import ServingPolicy
from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, eligibility_mask
from aep.synthetic.features import Standardizer, build_context_matrix
from aep.synthetic.generate import load_synthetic
from aep.synthetic.reward_model import RewardModel


@dataclass
class FairnessResult:
    exposure_by_segment: pd.DataFrame  # rows=segment, cols=offer share
    expected_conv_by_segment: pd.Series  # oracle expected conversion per segment
    value_disparity_ratio: float  # min/max expected conversion across segments
    top_offer_share: float  # largest single-offer exposure share overall


def evaluate_fairness(policy: ServingPolicy | None = None) -> FairnessResult:
    policy = policy or ServingPolicy().fit()
    base = load_processed()
    std = Standardizer.fit(base)
    rm = RewardModel.build()

    data = load_synthetic()
    events = data["offer_events"]
    clients = base.iloc[events["client_idx"].to_numpy()].reset_index(drop=True)
    X, _ = build_context_matrix(clients, std)
    elig = eligibility_mask(clients)
    sub = clients["subscribed"].to_numpy().astype(float)

    arms = policy.decide_indices(X, elig)
    chosen_offer = np.array([OFFER_IDS[a] for a in arms])
    segment = events["segment"].to_numpy()
    r_all = rm.reward_prob(X, sub)
    chosen_oracle_reward = r_all[np.arange(len(events)), arms]

    df = pd.DataFrame({"segment": segment, "offer": chosen_offer, "reward": chosen_oracle_reward})
    exposure = (
        pd.crosstab(df["segment"], df["offer"], normalize="index")
        .reindex(columns=OFFER_IDS, fill_value=0.0)
        .round(4)
    )
    expected_conv = df.groupby("segment")["reward"].mean().round(5)
    disparity = float(expected_conv.min() / expected_conv.max()) if expected_conv.max() > 0 else 0.0
    top_share = float(df["offer"].value_counts(normalize=True).iloc[0])

    return FairnessResult(
        exposure_by_segment=exposure,
        expected_conv_by_segment=expected_conv,
        value_disparity_ratio=disparity,
        top_offer_share=top_share,
    )
