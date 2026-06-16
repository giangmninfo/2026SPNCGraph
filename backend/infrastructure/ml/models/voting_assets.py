# backend/infrastructure/ml/models/voting_assets.py

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VotingAssets:
    features: np.ndarray     # (N, D)
    metadata: pd.DataFrame   # must include subject + grade