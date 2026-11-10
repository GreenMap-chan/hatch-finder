"""Transfer complete parameterized layers using semantic architecture locations."""
import re

import torch
from torch import nn

from .checkpoint import model_state_dict
from .config import model_config_from_pt_data


def _semantic_name(name, config):
    match = re.match(r"heatmap_decoder.blocks\.(\d+)(.*)", name)
    if match:
        return f"decoder.level.{len(config.drawing_channels) - 2 - int(match[1])}{match[2]}"
    if name.startswith(("heatmap_decoder.input_projection", "heatmap_decoder.input_film")):
        return f"deepest.{len(config.drawing_channels) - 1}.{name}"
    match = re.match(r"output_comparison.matchers\.(\d+)\.(\d+)(.*)", name)
    if match:
        index = int(match[2]) // 2
        role = "output" if index == len(config.matcher_channels) else f"hidden.{index}"
        return f"matcher.{match[1]}.{role}{match[3]}"
    return name


def _signature(name, config):
    # Equal tensor shapes alone do not establish semantic compatibility.
    match = re.fullmatch(r"hatch_projections\.(\d+)", name)
    if match:
        level = int(match[1])
        return config.hatch_channels[level], config.hatch_pool_sizes[level]
    match = re.match(r"heatmap_decoder.blocks\.(\d+)\.0$", name)
    if match:
        level = len(config.drawing_channels) - 2 - int(match[1])
        return (config.drawing_channels[level + 1], config.drawing_channels[level],
                config.match_feature_dims[level] if config.match_dims[level] else 0)
    if name.startswith("drawing_encoder.blocks"):
        return config.group_norm_groups_drawings if name.endswith((".1", ".4")) else None
    if name.startswith("hatch_encoder.blocks"):
        return config.group_norm_groups_hatchings if name.endswith((".1", ".4")) else None
    return None


def transfer_weights(model, path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    old_config = model_config_from_pt_data(data)
    if old_config is None:
        raise ValueError("Architecture transfer requires a .pt file with model configuration")
    new_config = model.config.model
    source = model_state_dict(data)
    destination = model.state_dict()
    old_by_semantic = {_semantic_name(key, old_config): key for key in source}
    consumed = set()
    copied = 0
    report = [f"Architecture transfer from: {path}"]
    old_dump, new_dump = old_config.model_dump(), new_config.model_dump()
    for key in new_dump:
        if old_dump[key] != new_dump[key]:
            report.append(f"Config {key}: {old_dump[key]} -> {new_dump[key]}")

    # MultiheadAttention owns its output projection too: reset it atomically.
    attention_prefixes = [name for name, module in model.named_modules() if isinstance(module, nn.MultiheadAttention)]
    groups = {}
    for key in destination:
        owner = next((prefix for prefix in attention_prefixes if key.startswith(prefix + ".")), key.rsplit(".", 1)[0])
        if key.startswith("heatmap_decoder.match_gates."):
            owner = key  # Each level has an independent scalar, not a shared layer.
        groups.setdefault(owner, []).append(key)

    for owner, keys in groups.items():
        mapping = {key: old_by_semantic.get(_semantic_name(key, new_config)) for key in keys}
        reason = None
        if any(value is None for value in mapping.values()):
            reason = "new layer or changed role"
        elif any(destination[key].shape != source[value].shape for key, value in mapping.items()):
            reason = "parameter shape changed"
        else:
            old_owner = next(iter(mapping.values())).rsplit(".", 1)[0]
            if _signature(owner, new_config) != _signature(old_owner, old_config):
                reason = "input layout or normalization changed"
        if owner in attention_prefixes:
            for field, prefix in (("transformers", "drawing_encoder.transformers."),
                                  ("hatch_transformers", "hatch_encoder.transformers."),
                                  ("cross_attention", "cross_attention.")):
                if owner.startswith(prefix):
                    level = int(owner[len(prefix):].split(".")[0])
                    old = {item.level: item for item in getattr(old_config, field)}.get(level)
                    new = {item.level: item for item in getattr(new_config, field)}[level]
                    if old is not None and old.num_heads != new.num_heads:
                        reason = "attention head count changed"
        if reason:
            report.append(f"Initialized {owner}: {reason}")
        else:
            for key, old_key in mapping.items():
                destination[key] = source[old_key]
                consumed.add(old_key)
                copied += destination[key].numel()
            report.append(f"Transferred {owner}")

    model.load_state_dict(destination)
    discarded = sorted({key.rsplit(".", 1)[0] for key in source if key not in consumed})
    if discarded:
        report.append("Unused source layers (removed or reinitialized): " + ", ".join(discarded))
    old_count = sum(value.numel() for value in source.values())
    new_count = sum(value.numel() for value in destination.values())
    report.append(f"Parameters: {old_count:,} -> {new_count:,}; transferred {copied:,}/{new_count:,} ({100 * copied / new_count:.2f}%)")
    if not copied:
        report.append("No weights transferred: training starts from scratch.")
    return report
