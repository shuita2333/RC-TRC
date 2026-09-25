"""Run greedy Qwen2.5-VL generation with a fixed TRC residual mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from residual_stream_utils import register_residual_bypass_masks
from trc_common import generate, load_model, prepare_input, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Request JSONL")
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="Checkpoint directory or model identifier")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = torch.load(args.mask, map_location="cpu", weights_only=False)
    if payload.get("format") != "trc_qwen25vl_attention_adaptive_v2":
        raise ValueError("Mask is not the expected TRC Qwen2.5-VL format")
    alpha = float(payload["alpha_l_star"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid mask suppression ratio: {alpha}")
    selected_layer = int(payload["selected_layers"][0])
    selected_dimensions = payload["selected_dimensions"]
    weights = payload["mask_weights"]["attention"]
    expected_retention = 1.0 - alpha
    if not torch.allclose(weights[selected_layer, selected_dimensions],
                          torch.full((len(selected_dimensions),), expected_retention)):
        raise ValueError("Mask weights do not match the computed suppression ratio")
    model, processor, tokenizer, layers = load_model(args.model_path, args.device)
    if len(layers) != payload["mask_weights"]["attention"].shape[0]:
        raise ValueError("Mask and model layer counts differ")
    handles = register_residual_bypass_masks(layers, payload, streams=("attention",))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("w", encoding="utf-8") as output:
            for number, row in read_jsonl(args.input):
                inputs = prepare_input(processor, row, args.input, args.device)
                ids, text = generate(model, tokenizer, inputs)
                output.write(json.dumps({
                    "source_line": number,
                    "new_generated_token_count": int(ids.numel()),
                    "generated_text": text,
                }, ensure_ascii=False) + "\n")
                output.flush()
                print(f"Defended request {number}: {ids.numel()} new tokens", flush=True)
    finally:
        for handle in reversed(handles):
            handle.remove()


if __name__ == "__main__":
    main()
