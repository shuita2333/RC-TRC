# Uncovering Uncontrolled Repetition through Residual Stream Dynamics

This repository contains the core implementation of **Tokenwise Residual Comparison (TRC)** for localizing repetition-related residual behavior and applying a fixed soft defense. The reference configuration uses Qwen2.5-VL-3B-Instruct and runs in three stages:

1. **Localize:** compare attention-input residual differences from attack-induced repetition and benign generations.
2. **Calibrate the mask:** select the attack-associated differences at that layer and compute their suppression strength from the localization result.
3. **Defend:** reuse the fixed mask during greedy generation on new requests.

This is the main-method code path. Model checkpoints, datasets, attack construction, generated outputs, and ablation experiments are not bundled.

## Repository layout

| File | Role |
| --- | --- |
| `localize.py` | Collect calibration vectors, score candidate layers, and compute the suppression ratio. |
| `train_mask.py` | Convert the selected layer and attack vectors into a fixed soft mask. |
| `defend.py` | Load the mask and generate defended responses. |
| `trc_common.py` | Qwen2.5-VL input preparation, generation, token pairing, and residual capture. |
| `residual_stream_utils.py` | Teacher-forced residual capture and the inference-time residual-bypass hook. |

## Environment

A CUDA-capable GPU is required for model calibration and defended generation. The code was checked on an NVIDIA RTX A4000 GPU with Python 3.10, PyTorch 2.5.0+cu124, Transformers 4.50.0, NumPy 1.26.4, Pillow 11.1.0, and Accelerate 1.1.0. Install a PyTorch build compatible with your GPU and CUDA runtime, then install the remaining pinned dependencies:

```bash
git clone git@github.com:shuita2333/RC-TRC.git
cd RC-TRC
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The PyTorch command above targets CUDA 12.4. For another CUDA runtime, choose the corresponding PyTorch 2.5.0 wheel from the [official version archive](https://docs.pytorch.org/get-started/previous-versions/).

Place the Qwen2.5-VL-3B-Instruct checkpoint under `models/Qwen2.5-VL-3B-Instruct`, or use a model identifier available in your environment. Pass the same `--model-path` to localization and defense. The path is used at runtime and is not written into calibration files or the mask. Run both stages from the repository root when using a relative path.

The residual hook checks the decoder's forward structure before applying a mask. Other Transformers versions or model architectures may need an adapted hook.

## Prepare inputs

Each stage reads UTF-8 JSONL: one JSON object per nonempty line. A multimodal request has this shape:

```json
{"prompt": "Describe the image.", "image_path": "images/example.png"}
```

Relative `image_path` values are resolved relative to the JSONL file. It may be omitted for text-only requests. The scripts use `prompt` verbatim; they do not infer prompts from attack folders or create adversarial examples.

Prepare three files:

| File | Content and selection rule |
| --- | --- |
| `attack.jsonl` | Calibration requests that induce uncontrolled repetition. |
| `normal.jsonl` | Benign calibration requests. |
| `requests.jsonl` | New requests for defended generation. Keep them separate from calibration requests. |

## Run the three stages

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0

python localize.py \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --attack data/attack.jsonl \
  --normal data/normal.jsonl \
  --output-dir output/calibration

python train_mask.py \
  --calibration-dir output/calibration \
  --output output/trc_mask.pt

python defend.py \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --input data/requests.jsonl \
  --mask output/trc_mask.pt \
  --output output/defended.jsonl
```

## Outputs

| Stage | Output | Meaning |
| --- | --- | --- |
| Localization | `output/calibration/localization.json` | Selected layer, layer curves and scores, `relative_amplitude_at_selected_layer` (`a_l_star`), `alpha_l_star` (the suppression ratio), and selected-coordinate mask value. |
| Localization | `output/calibration/calibration.npz` | Attack and benign direction-vector arrays, shaped `[samples, layers, hidden_size]`. |
| Mask training | `output/trc_mask.pt` | PyTorch payload with selected coordinates, computed suppression ratio, and full attention mask matrix. |
| Defense | `output/defended.jsonl` | One row per request with source line, generated-token count, and response text. |
