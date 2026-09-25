"""Build one fixed Top-5% soft mask from calibrated attack vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

FRACTION = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.calibration_dir / "localization.json").read_text(encoding="utf-8"))
    with np.load(args.calibration_dir / "calibration.npz") as data:
        attack = data["attack"]
    if attack.ndim != 3 or attack.shape[0] != metadata["attack_samples"]:
        raise ValueError("Invalid attack calibration array")
    layer = int(metadata["attention_start_layer"])
    alpha = float(metadata["alpha_l_star"])
    if not 0.0 <= alpha <= 1.0 or not np.isfinite(alpha):
        raise ValueError(f"Invalid computed suppression ratio: {alpha}")
    retention = 1.0 - alpha
    n_layers, hidden_size = attack.shape[1:]
    if not 0 <= layer < n_layers:
        raise ValueError("Localized layer outside mask shape")
    mean_directions = attack[:, layer, :].mean(axis=0)
    count = int(np.ceil(hidden_size * FRACTION))
    selected = np.argsort(-mean_directions, kind="stable")[:count]
    weights = torch.ones((n_layers, hidden_size), dtype=torch.float32)
    weights[layer, selected.tolist()] = retention
    payload = {
        "format": "trc_qwen25vl_attention_adaptive_v2",
        "module": "residual_bypass",
        "streams": ["attention"],
        "hidden_size": hidden_size,
        "selected_layers": [layer],
        "mask_fraction": FRACTION,
        "alpha_l_star": alpha,
        "suppression_ratio": alpha,
        "selected_dimension_retention": retention,
        "relative_amplitude_at_selected_layer": metadata["relative_amplitude_at_selected_layer"],
        "selected_dimensions": selected.tolist(),
        "mask_weights": {"attention": weights},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.output}: attention layer {layer}, {count} dimensions, suppression={alpha:.6f} ({alpha:.2%}), retention={retention:.6f}", flush=True)


if __name__ == "__main__":
    main()
