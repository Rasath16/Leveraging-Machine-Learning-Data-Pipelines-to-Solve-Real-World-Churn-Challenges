from __future__ import annotations
import os
import numpy as np
import joblib
from typing import Dict, Any

class ArtifactSaver:
    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    @staticmethod
    def save_npz(path: str, array) -> None:
        ArtifactSaver._ensure_dir(path)
        # Save as single-array NPZ for consistency
        np.savez(path, array)

    @staticmethod
    def save_npy(path: str, array) -> None:
        ArtifactSaver._ensure_dir(path)
        np.save(path, array)

    @staticmethod
    def save_preprocessor(path: str, preprocessor: Any) -> None:
        ArtifactSaver._ensure_dir(path)
        joblib.dump(preprocessor, path)
