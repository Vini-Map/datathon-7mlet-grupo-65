"""Off-policy evaluation (Stage 4).

Estimates the value (conversion rate) of the frozen serving policy from the
**logged** uniform-policy data, using inverse-propensity scoring (IPS) and its
self-normalized variant (SNIPS). Because the layer is synthetic we also compute
the policy's *true* value from the oracle, which validates the OPE estimates
(IPS/SNIPS should land near the oracle-true value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aep.bandits.serving import ServingPolicy
from aep.data.loader import load_processed
from aep.synthetic.catalog import OFFER_IDS, eligibility_mask
from aep.synthetic.features import Standardizer, build_context_matrix
from aep.synthetic.generate import load_synthetic
from aep.synthetic.reward_model import RewardModel


@dataclass
class OPEResult:
    logged_value: float  # observed conversion of the logging (uniform) policy
    ips: float  # IPS estimate of the target policy value
    snips: float  # self-normalized IPS estimate
    oracle_true_target: float  # true target value via the oracle (synthetic only)
    oracle_upper_bound: float  # best-eligible oracle value (optimal policy)
    effective_sample: float  # share of logged events where target == logged action


def evaluate_offpolicy(policy: ServingPolicy | None = None) -> OPEResult:
    policy = policy or ServingPolicy().fit()
    base = load_processed()
    std = Standardizer.fit(base)
    rm = RewardModel.build()

    data = load_synthetic()
    events = data["offer_events"].merge(
        data["delayed_rewards"][["event_id", "conversion"]], on="event_id"
    )
    clients = base.iloc[events["client_idx"].to_numpy()].reset_index(drop=True)
    X, _ = build_context_matrix(clients, std)
    elig = eligibility_mask(clients)
    sub = clients["subscribed"].to_numpy().astype(float)

    arm_of = {oid: i for i, oid in enumerate(OFFER_IDS)}
    logged_arm = np.array([arm_of[o] for o in events["offer_id"]])
    reward = events["conversion"].to_numpy().astype(float)
    propensity = events["propensity"].to_numpy()

    target_arm = policy.decide_indices(X, elig)
    match = target_arm == logged_arm

    # IPS / SNIPS.
    w = np.where(match, 1.0 / propensity, 0.0)
    ips = float(np.mean(w * reward))
    snips = float(np.sum(w * reward) / np.sum(w)) if w.sum() > 0 else 0.0

    # Oracle ground truth (synthetic): true value of target & optimal policies.
    r_all = rm.reward_prob(X, sub)
    rows = np.arange(len(events))
    oracle_true_target = float(r_all[rows, target_arm].mean())
    r_masked = np.where(elig, r_all, -1.0)
    oracle_upper = float(r_masked.max(axis=1).mean())

    return OPEResult(
        logged_value=float(reward.mean()),
        ips=ips,
        snips=snips,
        oracle_true_target=oracle_true_target,
        oracle_upper_bound=oracle_upper,
        effective_sample=float(match.mean()),
    )
