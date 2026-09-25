"""Calibrate attention TRC and select one intervention layer."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from trc_common import (
    generate, load_model, normal_pairs, prepare_input,
    read_jsonl, repeat_pairs, sample_vectors,
)

SAMPLE_COUNT = 5
WINDOW = 2


def collect(path, kind, model, processor, tokenizer, layers, device):
    vectors = []
    for number, row in read_jsonl(path):
        inputs = prepare_input(processor, row, path, device)
        ids, _ = generate(model, tokenizer, inputs)
        pairs = repeat_pairs(ids, tokenizer) if kind == "attack" else normal_pairs(ids, tokenizer)
        if pairs is None:
            print(f"Skip {kind} line {number}: insufficient eligible tokens", flush=True)
            continue
        vectors.append(sample_vectors(model, layers, inputs, ids, pairs))
        print(f"Accepted {kind} {len(vectors)}/{SAMPLE_COUNT}: line {number}", flush=True)
        if len(vectors) == SAMPLE_COUNT:
            break
    if len(vectors) != SAMPLE_COUNT:
        raise RuntimeError(f"{path}: found {len(vectors)} valid {kind} samples; need {SAMPLE_COUNT}")
    return np.stack(vectors)


def localize(attack, normal):
    """Strict C-RS-LGE: normal-relative magnitude times normalized growth."""
    attack_curve = attack.mean(axis=(0, 2), dtype=np.float64)
    normal_curve = normal.mean(axis=(0, 2), dtype=np.float64)
    energy_values = []
    for reference in itertools.combinations(range(len(normal)), 2):
        target = [i for i in range(len(normal)) if i not in reference]
        curve = normal[target].mean(axis=(0, 2), dtype=np.float64)
        energy_values.extend((((curve[WINDOW:] - curve[:-WINDOW]) / WINDOW) ** 2).tolist())
    positive = np.asarray([value for value in energy_values if value > 0], dtype=float)
    energy_scale = float(np.exp(np.log(positive).mean())) if len(positive) else float(np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.where(normal_curve[:-WINDOW] > 0,
                             attack_curve[:-WINDOW] / normal_curve[:-WINDOW],
                             np.where(attack_curve[:-WINDOW] == 0, 1.0, np.inf))
    growth = ((attack_curve[WINDOW:] - attack_curve[:-WINDOW]) / WINDOW) ** 2
    scores = amplitude * (1.0 + growth / energy_scale)
    scores[0] = np.inf  # Exclude attention layer zero.
    if not np.isfinite(scores).any():
        raise RuntimeError("No finite candidate layer score")
    selected = int(np.argmin(scores))
    return selected, attack_curve, normal_curve, scores, amplitude, energy_scale


def suppression_from_relative_amplitude(amplitude: float):
    """Top-1 suppression ratio: g=max(0,-ln(a)), alpha=g/(1+g)."""
    amplitude = float(amplitude)
    if np.isnan(amplitude) or amplitude < 0:
        raise ValueError(f"Invalid normal-relative amplitude: {amplitude}")
    if amplitude == 0:
        return float("inf"), 1.0
    if amplitude >= 1:
        return 0.0, 0.0
    gap = -float(np.log(amplitude))
    return gap, gap / (1.0 + gap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", type=Path, required=True, help="Attack calibration JSONL")
    parser.add_argument("--normal", type=Path, required=True, help="Benign calibration JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="Checkpoint directory or model identifier")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, processor, tokenizer, layers = load_model(args.model_path, args.device)
    attack = collect(args.attack, "attack", model, processor, tokenizer, layers, args.device)
    normal = collect(args.normal, "normal", model, processor, tokenizer, layers, args.device)
    selected, attack_curve, normal_curve, scores, amplitude, scale = localize(attack, normal)
    top1_amplitude = float(amplitude[selected])
    log_gap, alpha = suppression_from_relative_amplitude(top1_amplitude)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "calibration.npz", attack=attack, normal=normal)
    result = {
        "attention_start_layer": selected,
        "window": WINDOW,
        "attack_samples": SAMPLE_COUNT,
        "normal_samples": SAMPLE_COUNT,
        "normal_energy_scale": scale,
        "relative_amplitude_at_selected_layer": top1_amplitude,
        "suppression_log_gap": log_gap if np.isfinite(log_gap) else None,
        "alpha_l_star": alpha,
        "suppression_ratio": alpha,
        "selected_direction_mask_value": 1.0 - alpha,
        "suppression_formula": "g=max(0,-ln(a_l_star)); alpha_l_star=g/(1+g)",
        "attack_layer_curve": attack_curve.tolist(),
        "normal_layer_curve": normal_curve.tolist(),
        "layer_scores": [float(value) if np.isfinite(value) else None for value in scores],
    }
    (args.output_dir / "localization.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Selected attention layer {selected}; suppression alpha={alpha:.6f} ({alpha:.2%}); saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
