"""Drift and reward monitoring (Stage 7).

Two monitors over the logged stream, split into a reference and a current window:

- **Data drift:** Population Stability Index (PSI) per context feature.
- **Reward drift:** change in realized conversion rate between windows.

The natural synthetic stream is stationary (PSI ~ 0), which is the correct
baseline; ``monitor_drift(shock=...)`` can perturb the current window to show the
detectors fire, which is how we'd validate alerting before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aep.data.loader import load_processed
from aep.synthetic.features import CONTEXT_FEATURES, Standardizer, build_context_matrix
from aep.synthetic.generate import load_synthetic

PSI_WARN = 0.10  # moderate shift
PSI_ALERT = 0.25  # significant shift
REWARD_DRIFT_ALERT = 0.30  # relative change in conversion rate


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples (quantile bins)."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    max_psi: float
    drifted_features: list[str]
    reward_reference: float
    reward_current: float
    reward_relative_change: float
    reward_alert: bool

    @property
    def data_alert(self) -> bool:
        return self.max_psi >= PSI_ALERT


def _shock(X: np.ndarray, rewards: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the current window to validate the detectors (demo only)."""
    X = X.copy()
    rewards = rewards.copy()
    if kind in ("data", "both"):
        X[:, 0] += 1.5  # shift the (standardized) age feature
        X[:, 3] -= 1.0  # shift euribor3m
    if kind in ("reward", "both"):
        # halve the conversion signal in the current window
        keep = np.random.default_rng(0).random(len(rewards)) < 0.5
        rewards = rewards * keep
    return X, rewards


def monitor_drift(shock: str | None = None) -> DriftReport:
    """Compute data + reward drift between the first and second half of the stream."""
    base = load_processed()
    std = Standardizer.fit(base)
    data = load_synthetic()
    events = data["offer_events"].merge(
        data["delayed_rewards"][["event_id", "conversion"]], on="event_id"
    )
    clients = base.iloc[events["client_idx"].to_numpy()].reset_index(drop=True)
    X, _ = build_context_matrix(clients, std)
    rewards = events["conversion"].to_numpy().astype(float)

    split = np.median(events["day"].to_numpy())
    ref_mask = events["day"].to_numpy() < split
    cur_mask = ~ref_mask

    ref_X, cur_X = X[ref_mask], X[cur_mask]
    ref_r, cur_r = rewards[ref_mask], rewards[cur_mask]
    if shock:
        cur_X, cur_r = _shock(cur_X, cur_r, shock)

    feature_psi = {feat: psi(ref_X[:, i], cur_X[:, i]) for i, feat in enumerate(CONTEXT_FEATURES)}
    drifted = [f for f, v in feature_psi.items() if v >= PSI_WARN]
    rew_ref, rew_cur = float(ref_r.mean()), float(cur_r.mean())
    rel = (rew_cur - rew_ref) / rew_ref if rew_ref > 0 else 0.0

    return DriftReport(
        feature_psi=feature_psi,
        max_psi=max(feature_psi.values()),
        drifted_features=drifted,
        reward_reference=rew_ref,
        reward_current=rew_cur,
        reward_relative_change=rel,
        reward_alert=abs(rel) >= REWARD_DRIFT_ALERT,
    )
