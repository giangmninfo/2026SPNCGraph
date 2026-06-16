# backend/infrastructure/ml/models/voting_assets.py

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any


@dataclass(frozen=True)
class KNNGraphSAGEAssets:
    x: Any
    features: np.ndarray          # (N, D)
    metadata: pd.DataFrame        # subject, grade, etc.
    inv_label_map: Dict[int, str] # 0 -> "Công nghệ - 10"
