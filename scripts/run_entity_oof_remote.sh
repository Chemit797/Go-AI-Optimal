#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:?usage: run_entity_oof_remote.sh PROJECT_DIR [RUN_DIR]}"
RUN_DIR="${2:-runs/entity-oof-response-mvp-seed42}"
cd "$PROJECT_DIR"

export PYTHONPATH=src
mkdir -p "$RUN_DIR"

python - <<'PY'
import json
import platform
from datetime import datetime

import torch

payload = {
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "status": "running",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
with open("remote_environment.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
mv remote_environment.json "$RUN_DIR/remote_environment.json"

python -m goai_response.oof \
  --config configs/response_mvp.yaml \
  --run-dir "$RUN_DIR" \
  --n-folds 4 \
  --seed 42 \
  --resume \
  2>&1 | tee -a "$RUN_DIR/stdout.txt"
