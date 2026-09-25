"""Residual-stream capture and residual-bypass intervention utilities.

The supported decoder blocks (LLaMA/Vicuna used by InstructBLIP and LLaVA,
and Qwen2.5-VL) have the following pre-norm structure::

    h_attn = h + attention(norm(h))
    h_next = h_attn + mlp(norm(h_attn))

``attention`` below denotes the attention residual bypass ``h`` and ``mlp``
denotes the MLP residual bypass ``h_attn``.  They are deliberately *not* the
attention/MLP module output activations.
"""
from __future__ import annotations

from collections.abc import Mapping
import ast
import inspect
import textwrap
import types

import numpy as np
import torch


STREAMS = ("attention", "mlp")


def attention_module(layer):
    for name in ("self_attn", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError(f"Cannot find attention module in {type(layer).__name__}.")


def mlp_module(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError(f"Cannot find MLP module in {type(layer).__name__}.")


def post_attention_norm(layer):
    """Return the norm whose input is the unnormalised MLP residual stream."""
    for name in ("post_attention_layernorm", "post_attention_norm"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError(
        f"{type(layer).__name__} has no post-attention norm; cannot capture "
        "the pre-MLP residual stream safely."
    )


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output(output, value):
    return (value,) + output[1:] if isinstance(output, tuple) else value


def _teacher_forced_inputs(inputs, generated_ids):
    full_inputs = {key: value for key, value in inputs.items()}
    full_inputs["input_ids"] = torch.cat(
        (inputs["input_ids"], generated_ids.unsqueeze(0)), dim=1
    )
    if inputs.get("attention_mask") is not None:
        mask = inputs["attention_mask"]
        extension = torch.ones(
            (mask.shape[0], generated_ids.numel()), dtype=mask.dtype, device=mask.device
        )
        full_inputs["attention_mask"] = torch.cat((mask, extension), dim=1)
    return full_inputs


def capture_residual_matrices(model, layers, stream, inputs, generated_ids):
    """Capture one [sequence, hidden] residual-stream matrix per layer.

    Attention uses a decoder-layer pre-hook, therefore captures ``h_l``.
    MLP uses a pre-hook on ``post_attention_layernorm``, therefore captures
    ``h_l + a_l`` before normalisation, i.e. the vector used by the MLP
    residual addition.  The model runs once teacher-forced with cache off.
    """
    if stream not in STREAMS:
        raise ValueError(f"Unsupported residual stream: {stream}")
    matrices, handles = {}, []
    for layer_index, layer in enumerate(layers):
        target = layer if stream == "attention" else post_attention_norm(layer)

        def hook(_module, args, index=layer_index):
            if not args:
                raise RuntimeError("Residual stream hook received no positional hidden_states.")
            hidden_states = args[0]
            if hidden_states.ndim != 3:
                raise RuntimeError(
                    f"Residual stream hook expected [batch, sequence, hidden], "
                    f"got {tuple(hidden_states.shape)}."
                )
            if hidden_states.shape[0] != 1:
                raise RuntimeError(
                    f"Residual-stream capture requires batch size 1, got {hidden_states.shape[0]}."
                )
            # The experiment runs a single teacher-forced continuation.  Drop
            # its batch dimension so downstream indexing is [sequence, hidden]
            # rather than accidentally treating batch size (1) as sequence.
            matrices[index] = hidden_states[0].detach().float().cpu().numpy()

        handles.append(target.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(**_teacher_forced_inputs(inputs, generated_ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(matrices) != len(layers):
        raise RuntimeError(
            f"Captured {len(matrices)}/{len(layers)} matrices for {stream} residual stream."
        )
    return matrices


def aligned_generated_states(matrices, num_generated_tokens):
    """Map generated token j to its causal prediction-state row."""
    result = {}
    for layer, matrix in matrices.items():
        if matrix.shape[0] < num_generated_tokens + 1:
            raise ValueError(
                f"Layer {layer} has {matrix.shape[0]} rows, need at least "
                f"{num_generated_tokens + 1} for generated-token alignment."
            )
        result[layer] = matrix[-(num_generated_tokens + 1):-1]
    return result


def scores_for_pairs(states, pairs):
    left = [left for left, _ in pairs]
    right = [right for _, right in pairs]
    scores = {}
    for layer, matrix in states.items():
        delta = matrix[right] - matrix[left]
        scores[layer] = {
            "layer_score": float(np.mean(np.abs(delta))),
            "absolute_dimension_mean": np.mean(np.abs(delta), axis=0),
            "signed_dimension_mean": np.mean(delta, axis=0),
        }
    return scores


def capture_residual_scores(model, layers, inputs, generated_ids, pairs, streams=STREAMS):
    """Capture absolute residual-stream differences for the requested pairs."""
    output = {}
    for stream in streams:
        matrices = capture_residual_matrices(model, layers, stream, inputs, generated_ids)
        output[stream] = scores_for_pairs(
            aligned_generated_states(matrices, len(generated_ids)), pairs
        )
    return output


def _normalise_layer_weights(mask_weights, hidden_size):
    output = {}
    for raw_layer, raw_weight in mask_weights.items():
        layer = int(raw_layer)
        weight = torch.as_tensor(raw_weight, dtype=torch.float32).flatten().cpu()
        if weight.numel() != hidden_size:
            raise ValueError(
                f"Layer {layer} mask has {weight.numel()} dimensions; expected {hidden_size}."
            )
        output[layer] = weight
    return output


def full_mask_matrix(weights, num_layers, hidden_size):
    matrix = torch.ones(num_layers, hidden_size, dtype=torch.float32)
    if isinstance(weights, Mapping):
        for index, row in _normalise_layer_weights(weights, hidden_size).items():
            if not 0 <= index < num_layers:
                raise ValueError(f"Invalid mask layer: {index}")
            matrix[index] = row
    else:
        matrix = torch.as_tensor(weights).detach().float().cpu().clone()
        if matrix.shape != (num_layers, hidden_size):
            raise ValueError(f"Invalid full mask shape: {matrix.shape}")
    if not torch.isfinite(matrix).all() or not torch.all((matrix >= 0) & (matrix <= 1)):
        raise ValueError("Residual-mask weights must be finite values in the closed interval [0, 1].")
    return matrix


def _masked_residual(layer, residual, stream):
    weight = layer._direct_residual_weights.get(stream)
    if weight is None:
        return residual
    return residual * weight.to(device=residual.device, dtype=residual.dtype).view(1, 1, -1)


def _direct_forward(layer):
    """Preserve the installed model's forward, changing only its two additions.

    Compile a private copy of the installed decoder forward. Attention calls,
    MLP calls, cache handling, arguments and return structure are preserved.
    Only sequential pre-norm decoders with two residual additions are accepted.
    """
    original = layer.forward
    function = original.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    definition.decorator_list = []
    additions = []
    for statement in definition.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == 'hidden_states'
            and isinstance(statement.value, ast.BinOp)
            and isinstance(statement.value.op, ast.Add)
            and isinstance(statement.value.left, ast.Name)
            and statement.value.left.id == 'residual'
            and isinstance(statement.value.right, ast.Name)
            and statement.value.right.id == 'hidden_states'):
            additions.append(statement)
    if len(additions) != 2 or not hasattr(layer, 'post_attention_layernorm'):
        raise ValueError(f"Unsupported decoder residual structure: {type(layer).__name__}")
    for statement, stream, output_name in zip(additions, STREAMS, ('attention_output', 'mlp_output')):
        index = definition.body.index(statement)
        replacements = ast.parse(
            f"{output_name} = hidden_states\n"
            f"masked_residual = _direct_mask_residual(self, residual, {stream!r})\n"
            f"hidden_states = masked_residual + {output_name}\n"
        ).body
        definition.body[index:index + 1] = replacements
    namespace = dict(function.__globals__)
    namespace['_direct_mask_residual'] = _masked_residual
    exec(compile(ast.fix_missing_locations(tree), '<direct_residual_forward>', 'exec'), namespace)
    modified = namespace[definition.name]
    modified.__defaults__ = function.__defaults__
    modified.__kwdefaults__ = function.__kwdefaults__
    return types.MethodType(modified, layer)


class _ResidualHandle:
    def __init__(self, layer, stream):
        self.layer, self.stream = layer, stream

    def remove(self):
        if self.layer is None:
            return
        layer = self.layer
        layer._direct_residual_weights.pop(self.stream)
        if not layer._direct_residual_weights:
            original, had_override = layer._direct_residual_original
            if had_override:
                layer.forward = original
            else:
                del layer.forward
            del layer._direct_residual_original
            del layer._direct_residual_weights
        self.layer = None


def register_residual_bypass_masks(layers, mask_payload, streams=None):
    """Directly multiply residual before addition; module outputs stay untouched."""
    if mask_payload.get('module') != 'residual_bypass':
        raise ValueError('Expected residual_bypass payload.')
    requested = tuple(streams if streams is not None else mask_payload['streams'])
    if len(set(requested)) != len(requested) or not set(requested) <= set(STREAMS):
        raise ValueError('Invalid or duplicate streams.')
    matrices = {stream: full_mask_matrix(mask_payload['mask_weights'][stream],
                len(layers), int(mask_payload['hidden_size'])) for stream in requested}
    handles = []
    try:
        for index, layer in enumerate(layers):
            for stream in requested:
                weight = matrices[stream][index]
                if torch.all(weight == 1):
                    continue
                if not hasattr(layer, '_direct_residual_weights'):
                    modified = _direct_forward(layer)
                    layer._direct_residual_original = (layer.forward, 'forward' in layer.__dict__)
                    layer._direct_residual_weights = {}
                    layer.forward = modified
                if stream in layer._direct_residual_weights:
                    raise ValueError(f'Mask already registered for layer {index}/{stream}')
                layer._direct_residual_weights[stream] = weight
                handles.append(_ResidualHandle(layer, stream))
    except Exception:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles
