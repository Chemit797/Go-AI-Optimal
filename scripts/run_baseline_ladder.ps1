param(
    [string]$Config = "configs/baseline.yaml",
    [switch]$IncludeP1ToP4
)

$ErrorActionPreference = "Stop"

python -m goai_baseline.audit --config $Config
python -m goai_baseline.preprocess --config $Config --output "runs/preprocess/feature_contract.json"
python -m goai_baseline.evaluate --config $Config --variant b0_mean --output-dir "runs/b0_mean"
python -m goai_baseline.evaluate --config $Config --variant b1_matched_control --output-dir "runs/b1_matched_control"
python -m goai_baseline.train --config $Config --variant p0_onehot

if ($IncludeP1ToP4) {
    python -m goai_baseline.train --config $Config --variant p1_priors
    python -m goai_baseline.train --config $Config --variant p2_crosses
    python -m goai_baseline.train --config $Config --variant p3_time
    python -m goai_baseline.train --config $Config --variant p4_hash
}
