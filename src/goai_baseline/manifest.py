"""Run metadata written alongside every local experiment."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .audit import audit_inputs
from .config import BaselineConfig


def _normalise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def write_manifest(path: str | Path, config: BaselineConfig, extra: dict[str, Any]) -> None:
    output = Path(path)
    with config.path.open("r", encoding="utf-8") as handle:
        config_payload = yaml.safe_load(handle)
    payload = {
        "config": config_payload,
        "input_audit": audit_inputs(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        **_normalise(extra),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
