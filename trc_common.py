"""Shared Qwen2.5-VL input, generation, and token-pair helpers for TRC."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from residual_stream_utils import capture_residual_scores

MAX_NEW_TOKENS = 2048
VISIBLE_START = 15
REPEAT_DISTANCE = 3
PAIR_TOKENS = 20


def load_model(model_path: str, device: str = "cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    layers = model.model.layers
    return model, processor, processor.tokenizer, layers


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{number}: expected an object with a prompt string")
            yield number, row


def prepare_input(processor, row: dict, source_path: Path, device: str):
    prompt = row["prompt"]
    image_value = row.get("image_path")
    if image_value:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = source_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    else:
        image = None
        content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if image is not None:
        kwargs["images"] = [image]
    return processor(**kwargs).to(device)


def generate(model, tokenizer, inputs):
    prompt_ids = inputs["input_ids"][0]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]
    if output.numel() < prompt_ids.numel() or not torch.equal(output[:prompt_ids.numel()], prompt_ids):
        raise RuntimeError("Could not separate prompt tokens from generated tokens")
    ids = output[prompt_ids.numel():]
    return ids, tokenizer.decode(ids, skip_special_tokens=True)


def visible_rows(ids, tokenizer):
    special = set(tokenizer.all_special_ids or [])
    rows = []
    for index, token in enumerate(ids.detach().cpu().tolist()):
        token = int(token)
        if token in special:
            continue
        text = tokenizer.decode([token], skip_special_tokens=True)
        if text.strip():
            rows.append((index, token, text))
    return rows


def repeat_pairs(ids, tokenizer):
    """Return 19 links from a 20-token visible repeat chain, or None."""
    visible = visible_rows(ids, tokenizer)[VISIBLE_START - 1:]
    for start in range(len(visible)):
        chain = [start]
        while len(chain) < PAIR_TOKENS:
            current = chain[-1]
            stop = min(len(visible), current + REPEAT_DISTANCE + 1)
            next_index = next(
                (i for i in range(current + 1, stop)
                 if visible[i][1] == visible[current][1]), None
            )
            if next_index is None:
                break
            chain.append(next_index)
        if len(chain) == PAIR_TOKENS:
            positions = [visible[index][0] for index in chain]
            return list(zip(positions[:-1], positions[1:]))
    return None


def normal_pairs(ids, tokenizer):
    visible = visible_rows(ids, tokenizer)[:PAIR_TOKENS]
    if len(visible) < PAIR_TOKENS:
        return None
    positions = [row[0] for row in visible]
    return list(zip(positions[:-1], positions[1:]))


def sample_vectors(model, layers, inputs, ids, pairs):
    scores = capture_residual_scores(
        model, layers, inputs, ids, pairs, streams=("attention",)
    )["attention"]
    return np.stack(
        [scores[index]["absolute_dimension_mean"] for index in range(len(layers))]
    ).astype(np.float32)
