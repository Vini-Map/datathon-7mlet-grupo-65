"""Constants and seeds for the synthetic experimentation layer (Stage 2).

Every random draw in this package derives from :data:`MASTER_SEED` through an
explicit ``numpy.random.default_rng`` generator, so the entire synthetic layer
is bit-for-bit reproducible and the seeds are documented in one place.
"""

from __future__ import annotations

from aep.config import REPO_ROOT, get_settings

# Master seed comes from settings (AEP_RANDOM_SEED, default 42). Sub-seeds below
# are offsets so independent components never share a stream by accident.
MASTER_SEED: int = get_settings().random_seed
SEED_REWARD_MODEL: int = MASTER_SEED + 101  # ground-truth reward parameters
SEED_EVENTS: int = MASTER_SEED + 202  # logged impressions / clients / actions
SEED_DELAY: int = MASTER_SEED + 303  # delayed-reward realization

# Simulation sizing.
N_OFFERS: int = 8
N_EVENTS: int = 20_000  # logged impressions
HORIZON_DAYS: int = 90  # temporal horizon over which events are spread

# Output locations (Parquet artifacts are git-ignored; schemas/reports are not).
SYNTH_DIR = REPO_ROOT / "data" / "synthetic_enrichment"
OFFER_CATALOG_PATH = SYNTH_DIR / "offer_catalog.parquet"
OFFER_EVENTS_PATH = SYNTH_DIR / "offer_events.parquet"
DELAYED_REWARDS_PATH = SYNTH_DIR / "delayed_rewards.parquet"
REWARD_PARAMS_PATH = SYNTH_DIR / "reward_model_params.json"
STANDARDIZER_PATH = SYNTH_DIR / "context_standardizer.json"

# Reward observation horizon: a non-converting impression is censored (reward
# known to be 0) after this many days; converters realize after a sampled delay.
CENSOR_WINDOW_DAYS: int = 14
