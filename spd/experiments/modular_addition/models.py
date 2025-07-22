"""Model definitions for modular addition experiments.

Vendored from: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper
"""

from pathlib import Path
from typing import Any

import torch

from .vendored.transformers import Config, Transformer, gen_train_test


def convert_state_dict(old_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert layers from nn.Parameter to nn.Embedding/nn.Linear format."""
    new_state_dict = {}
    
    for key, tensor in old_state_dict.items():
        if key == "embed.W_E":
            # Convert from (d_model, d_vocab) to (d_vocab, d_model) for nn.Embedding
            new_state_dict["embed.embedding.weight"] = tensor.T
        elif key == "unembed.W_U":
            # Convert from (d_model, d_vocab) to (d_vocab, d_model) for nn.Linear
            new_state_dict["unembed.linear.weight"] = tensor.T
        elif key == "pos_embed.W_pos":
            # Convert positional embedding: (max_ctx, d_model) -> (max_ctx, d_model) for nn.Embedding
            new_state_dict["pos_embed.pos_embedding.weight"] = tensor
        elif key.endswith(".W_in"):
            # MLP input layer: (d_mlp, d_model) -> (d_mlp, d_model) 
            new_key = key.replace(".W_in", ".W_in.weight")
            new_state_dict[new_key] = tensor
        elif key.endswith(".b_in"):
            # MLP input bias
            new_key = key.replace(".b_in", ".W_in.bias")
            new_state_dict[new_key] = tensor
        elif key.endswith(".W_out"):
            # MLP output layer: (d_model, d_mlp) -> (d_model, d_mlp)
            new_key = key.replace(".W_out", ".W_out.weight")
            new_state_dict[new_key] = tensor
        elif key.endswith(".b_out"):
            # MLP output bias
            new_key = key.replace(".b_out", ".W_out.bias")
            new_state_dict[new_key] = tensor
        elif key.endswith(".W_Q"):
            # Attention Q weights: (num_heads, d_head, d_model) -> separate per head
            layer_prefix = key.replace(".W_Q", "")
            num_heads, d_head, d_model = tensor.shape
            for head_idx in range(num_heads):
                head_weight = tensor[head_idx]  # (d_head, d_model)
                new_key = f"{layer_prefix}.heads.{head_idx}.q_proj.weight"
                new_state_dict[new_key] = head_weight
        elif key.endswith(".W_K"):
            # Attention K weights: (num_heads, d_head, d_model) -> separate per head  
            layer_prefix = key.replace(".W_K", "")
            num_heads, d_head, d_model = tensor.shape
            for head_idx in range(num_heads):
                head_weight = tensor[head_idx]  # (d_head, d_model)
                new_key = f"{layer_prefix}.heads.{head_idx}.k_proj.weight"
                new_state_dict[new_key] = head_weight
        elif key.endswith(".W_V"):
            # Attention V weights: (num_heads, d_head, d_model) -> separate per head
            layer_prefix = key.replace(".W_V", "")
            num_heads, d_head, d_model = tensor.shape
            for head_idx in range(num_heads):
                head_weight = tensor[head_idx]  # (d_head, d_model)
                new_key = f"{layer_prefix}.heads.{head_idx}.v_proj.weight"
                new_state_dict[new_key] = head_weight
        elif key.endswith(".W_O"):
            # Attention output weights: (d_model, num_heads * d_head) -> separate per head
            layer_prefix = key.replace(".W_O", "")
            d_model, total_d_head = tensor.shape
            d_head = total_d_head // 4  # Assuming 4 heads, could make this configurable
            for head_idx in range(4):
                # Extract this head's portion - need (d_model, d_head) for nn.Linear(d_head, d_model)
                start_idx = head_idx * d_head
                end_idx = (head_idx + 1) * d_head
                head_weight = tensor[:, start_idx:end_idx]  # (d_model, d_head) - correct for nn.Linear
                new_key = f"{layer_prefix}.heads.{head_idx}.o_proj.weight"
                new_state_dict[new_key] = head_weight
        else:
            # Keep all other keys unchanged
            new_state_dict[key] = tensor
    
    return new_state_dict


def load_pretrained_modular_addition_model(
    checkpoint_path: str | Path,
    epoch: int = 40000,
    device: str = "cuda"
) -> tuple[Transformer, dict[str, Any]]:
    """Load a pretrained modular addition model from checkpoint."""
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    
    data = torch.load(checkpoint_path, map_location=device)
    config_dict = data['config']
    config = Config(**config_dict)
    model = Transformer(config)
    
    epoch_index = epoch // 100
    if epoch_index >= len(data['state_dicts']):
        raise ValueError(f"Epoch {epoch} not available. Max epoch: {(len(data['state_dicts']) - 1) * 100}")
    
    old_state_dict = data['state_dicts'][epoch_index]
    new_state_dict = convert_state_dict(old_state_dict)
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, config_dict


def create_modular_addition_dataset(config: Config):
    """Create train/test split for modular addition dataset."""
    train_data, test_data = gen_train_test(config)
    
    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    test_tensor = torch.tensor(test_data, dtype=torch.long)
    
    return train_tensor, test_tensor