"""Plotting utilities for modular addition experiments."""

import torch
from matplotlib import pyplot as plt

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.utils.component_utils import calc_causal_importances


def create_modular_addition_plot_results(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, GateMLP | VectorGateMLP],
    device: str | torch.device,
    p: int = 113,
    **_,
) -> dict[str, plt.Figure]:
    """Create plotting results for modular addition decomposition experiments.

    Creates one comprehensive heatmap showing component activations for all 2×p inputs:
    - First p inputs: (0, 0), (0, 1), ..., (0, p-1) 
    - Next p inputs: (0, 0), (1, 0), ..., (p-1, 0)

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        device: Device to use
        p: Prime modulus for modular arithmetic (default: 113)
        **_: Additional keyword arguments (ignored)

    Returns:
        Dictionary of figures showing component activations
    """
    fig_dict = {}
    
    # Create combined input batch: all 2×p inputs
    all_inputs = []
        
    # First p inputs: b=0, a varies from 0 to p-1  
    for a in range(p):
        all_inputs.append([a, 0, p])

    # next p inputs: a=p-1, b varies from 0 to p-1
    for b in range(p):
        all_inputs.append([p-1, b, p])
    
    # Convert to tensor
    all_batch = torch.tensor(all_inputs, dtype=torch.long, device=device)
    
    # Forward pass to get pre-weight activations for all inputs
    _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        all_batch, module_names=list(components.keys())
    )
    
    # Get V matrices from components
    Vs = {module_name: v.V for module_name, v in components.items()}

    # Calculate causal importances for all inputs (only need standard version)
    ci_raw, _ = calc_causal_importances(
        pre_weight_acts=pre_weight_acts, 
        Vs=Vs, 
        gates=gates, 
        detach_inputs=False
    )
    
    # Group components by type, splitting attention by head
    def group_components(ci_dict):
        embeddings = {}
        attention_heads = {0: {}, 1: {}, 2: {}, 3: {}}  # 4 attention heads
        mlps_output = {}
        
        for name, ci in ci_dict.items():
            clean_name = name.replace("components.", "").replace("-", ".")
            
            if any(embed in name for embed in ["embed.embedding", "pos_embed.pos_embedding"]):
                embeddings[clean_name] = ci
            elif any(attn in name for attn in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
                # Extract head number from the component name
                # e.g., "blocks.0.attn.heads.2.q_proj" -> head 2
                if ".heads." in name:
                    head_num = int(name.split(".heads.")[1].split(".")[0])
                    if head_num in attention_heads:
                        attention_heads[head_num][clean_name] = ci
            elif any(mlp in name for mlp in [".W_in", ".W_out", "unembed.linear"]):
                mlps_output[clean_name] = ci
                
        return embeddings, attention_heads, mlps_output
    
    # Group components (only using standard causal importances)
    emb_raw, attn_heads_raw, mlp_raw = group_components(ci_raw)
    
    title = f"Modular Addition: (0:{p-1}, 0) + ({p-1}, 0:{p-1})"
    position_titles = ["Pos 0 (Op A)", "Pos 1 (Op B)", "Pos 2 (Output)"]
    
    def create_position_grid_figure(ci_dict, group_name):
        """Create a figure with components × positions grid."""
        if not ci_dict:
            return None
            
        n_components = len(ci_dict)
        n_positions = 3
        
        fig, axs = plt.subplots(
            n_components, n_positions,
            figsize=(5 * n_positions, 4 * n_components),
            constrained_layout=True,
            squeeze=False,
            dpi=150
        )
        
        images = []
        
        for comp_idx, (comp_name, ci_tensor) in enumerate(ci_dict.items()):
            for pos_idx in range(n_positions):
                ax = axs[comp_idx, pos_idx]
                
                # Ensure we have the position dimension
                if ci_tensor.ndim != 3:
                    raise ValueError(f"Expected 3D tensor (batch, seq_len, C) for component {comp_name}, got {ci_tensor.ndim}D")
                
                # Extract position-specific data
                mask_data = ci_tensor[:, pos_idx, :].detach().cpu().numpy()
                
                # Plot heatmap with fixed colorbar range
                im = ax.matshow(mask_data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
                images.append(im)
                
                # Formatting
                ax.xaxis.tick_bottom()
                ax.xaxis.set_label_position("bottom")
                ax.set_xlabel("Component index")
                
                # Set titles
                if pos_idx == 0:  # First column gets component name
                    clean_name = comp_name.replace("components.", "").replace("-", ".")
                    ax.set_ylabel(f"{clean_name}\nInput index")
                else:
                    ax.set_ylabel("Input index")
                
                if comp_idx == 0:  # Top row gets position titles
                    ax.set_title(position_titles[pos_idx])
        
        # Add unified colorbar with fixed range
        fig.colorbar(images[0], ax=axs.ravel().tolist(), shrink=0.8)
        
        # Overall figure title
        fig.suptitle(f"{title} - {group_name}", fontsize=16)
        
        return fig
    
    # Create figures for embeddings and MLPs
    if emb_raw:
        fig_dict["causal_importances_embeddings"] = create_position_grid_figure(emb_raw, "Embeddings")
    if mlp_raw:
        fig_dict["causal_importances_mlps_output"] = create_position_grid_figure(mlp_raw, "MLPs & Output")
    
    # Create separate figures for each attention head
    for head_num, head_components in attn_heads_raw.items():
        if head_components:  # Only create figure if head has components
            fig_dict[f"causal_importances_attention_head_{head_num}"] = create_position_grid_figure(
                head_components, f"Attention Head {head_num}"
            )
        
    return fig_dict


def create_modular_addition_metrics(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, GateMLP | VectorGateMLP],
    causal_importances: dict[str, torch.Tensor],
    device: str | torch.device,
    config,
    batch_size: int = 1024,
    **_,
) -> dict[str, float | int]:
    """Create modular addition task performance metrics.
    
    Args:
        model: The ComponentModel
        components: Dictionary of components  
        gates: Dictionary of gates
        causal_importances: Current causal importances
        device: Device to use
        config: SPD configuration object containing sigmoid_type
        batch_size: Number of examples to evaluate (for efficiency)
        **_: Additional arguments (ignored - p, seed, frac_train are hardcoded)
        
    Returns:
        Dictionary of performance metrics
    """
    from .vendored.transformers import Config as GrokkingConfig
    from .vendored.transformers import gen_train_test
    
    metrics = {}
    
    # Hardcoded hyperparameters for consistency
    p = 113
    seed = 0
    frac_train = 0.3
    
    # Create train/test data splits with hardcoded parameters
    grokking_config = GrokkingConfig(p=p, frac_train=frac_train, seed=seed)
    train_data, test_data = gen_train_test(grokking_config)
    
    def evaluate_accuracy(data, data_name):
        """Evaluate accuracy on a dataset split."""
        if len(data) == 0:
            return {}
            
        # Sample data for efficiency
        sample_data = data[:batch_size] if len(data) > batch_size else data
        batch = torch.tensor(sample_data, dtype=torch.long, device=device)
        
        # Run inference on the sample
        with torch.no_grad():
            # Get target model predictions
            target_logits = model.model(batch)
            target_preds = target_logits.argmax(dim=-1)[:, -1]  # Final position predictions
            
            # Get component model predictions (unmasked - should match target)
            unmasked_logits = model.forward_with_components(batch, components=components, masks=None)  
            unmasked_preds = unmasked_logits.argmax(dim=-1)[:, -1]
            
            # Get component model predictions (masked with current causal importances)
            target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                batch, module_names=list(components.keys())
            )
            Vs = {module_name: components[module_name].V for module_name in components}

            causal_importances, causal_importances_upper_leaky = calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                Vs=Vs,
                gates=gates,
                detach_inputs=False,
                sigmoid_type=config.sigmoid_type,
            )

            masked_logits = model.forward_with_components(batch, components=components, masks=causal_importances)
            masked_preds = masked_logits.argmax(dim=-1)[:, -1]
        
        # Calculate ground truth answers
        correct_answers = []
        for x, y, _ in batch:
            correct_answers.append((x.item() + y.item()) % p)
        correct_answers = torch.tensor(correct_answers, device=device)
        
        # Calculate accuracies
        target_acc = (target_preds == correct_answers).float().mean().item()
        unmasked_acc = (unmasked_preds == correct_answers).float().mean().item()  
        masked_acc = (masked_preds == correct_answers).float().mean().item()
        
        # Calculate cross-entropy losses (more continuous metric)
        import torch.nn.functional as F
        target_ce = F.cross_entropy(target_logits[:, -1, :], correct_answers).item()
        unmasked_ce = F.cross_entropy(unmasked_logits[:, -1, :], correct_answers).item()
        masked_ce = F.cross_entropy(masked_logits[:, -1, :], correct_answers).item()
        
        return {
            f"modular_add/{data_name}_target_accuracy": target_acc,
            f"modular_add/{data_name}_unmasked_accuracy": unmasked_acc,
            f"modular_add/{data_name}_masked_accuracy": masked_acc,
            f"modular_add/{data_name}_target_ce_loss": target_ce,
            f"modular_add/{data_name}_unmasked_ce_loss": unmasked_ce,
            f"modular_add/{data_name}_masked_ce_loss": masked_ce,
        }
    
    # Evaluate on both splits
    metrics.update(evaluate_accuracy(train_data, "train"))
    metrics.update(evaluate_accuracy(test_data, "test"))
    
    return metrics


