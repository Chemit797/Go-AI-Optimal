"""Re-score saved S1 fold models after a whole-drug fingerprint derangement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import load_config, write_json
from .models import ChemicalEncoder, ProteinDeltaModel
from .train_s1 import (
    fold_context_reference,
    fold_diagnostics,
    permute_validation_fingerprints,
    predict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    cache_path = Path(config["paths"]["private_cache"]) / "goai_s1_delta.npz"
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    output = (
        Path(config["paths"]["output_root"])
        / "logs"
        / "sensitivity"
        / f"{args.arm}_seed_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(4):
        directory = (
            Path(config["paths"]["output_root"])
            / "oof"
            / args.arm
            / f"seed_{args.seed}"
            / f"fold_{fold}"
        )
        model_payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        rna_config = config["rna_pretraining"]
        s1_config = config["goai_s1"]
        encoder = ChemicalEncoder(
            int(config["fingerprint"]["n_bits"]),
            int(rna_config["encoder_hidden"]),
            int(rna_config["encoder_dim"]),
            float(rna_config["dropout"]),
        )
        model = ProteinDeltaModel(
            encoder,
            [int(value) for value in arrays["context_cardinalities"]],
            int(rna_config["encoder_dim"]),
            int(s1_config["context_dim"]),
            int(s1_config["fusion_hidden"]),
            len(arrays["proteins"]),
            float(s1_config["dropout"]),
        ).to(device)
        model.load_state_dict(model_payload["model"], strict=True)
        valid = arrays["folds"] == fold
        train = ~valid
        permuted = permute_validation_fingerprints(
            arrays["fingerprints"].astype(np.float32), arrays["chemicals"], valid
        )
        prediction_z = predict(
            model,
            permuted,
            arrays["contexts"][valid].astype(np.int64),
            device,
            int(s1_config["batch_size"]),
        )
        prediction = prediction_z * model_payload["target_scale"] + model_payload["target_mean"]
        reference, reference_mask = fold_context_reference(
            arrays["delta"], arrays["mask"], arrays["context_keys"].astype(str), train, valid
        )
        metrics = fold_diagnostics(
            prediction, arrays["delta"][valid], arrays["mask"][valid], reference, reference_mask
        )
        rows.append({"arm": args.arm, "seed": args.seed, "fold": fold, **metrics})
        np.savez_compressed(
            output / f"fold_{fold}_predictions_permuted_chemical.npz",
            sample_ids=arrays["sample_ids"][valid],
            chemicals=arrays["chemicals"][valid],
            proteins=arrays["proteins"],
            pred_delta=prediction.astype(np.float32),
            fold=np.asarray([fold], dtype=np.int64),
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fold_metrics_permuted_chemical.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "protocol": "same_saved_model_whole_heldout_drug_fingerprint_derangement",
            "arm": args.arm,
            "seed": args.seed,
            "fold_metrics": frame.to_dict(orient="records"),
        },
    )
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
