#!/usr/bin/env python3
"""Test script to verify modular addition model loading and component attachment."""

import sys
from pathlib import Path

import torch

# Add SPD to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from spd.experiments.modular_addition.models import load_pretrained_modular_addition_model
from spd.experiments.modular_addition.vendored.transformers import Config as GrokkingConfig
from spd.experiments.modular_addition.vendored.transformers import gen_train_test
from spd.models.component_model import ComponentModel


def test_model_loading():
    """Test loading the pretrained modular addition model."""
    print("=== Testing Model Loading ===")
    
    checkpoint_path = Path(__file__).parent / "full_run_data.pth"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Run: python download_checkpoint.py")
        return None, None
    
    try:
        model, config_dict = load_pretrained_modular_addition_model(
            checkpoint_path=checkpoint_path,
            epoch=40000,
            device="cpu"  # Use CPU for testing
        )
        print("✅ Model loaded successfully")
        print(f"   Config: {config_dict}")
        print(f"   Model type: {type(model)}")
        return model, config_dict
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def test_model_inference(model, config_dict):
    """Test running inference on the model."""
    print("\n=== Testing Model Inference ===")
    
    if model is None:
        print("❌ No model to test")
        return False
    
    try:
        # Create some test data
        grokking_config = GrokkingConfig(**config_dict)
        train_data, _ = gen_train_test(grokking_config)
        
        # Take a small batch
        batch = torch.tensor(train_data[:8], dtype=torch.long)
        print(f"   Test batch shape: {batch.shape}")
        print(f"   Sample inputs: {batch[:2]}")
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(batch)
        
        print("✅ Inference successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check predictions for the sample inputs
        logits = output[:8, -1, :-1]  # Final position, excluding padding token
        predictions = logits.argmax(dim=-1)
        
        for i in range(8):
            x, y, _ = batch[i]
            expected = grokking_config.fn(x.item(), y.item())
            predicted = predictions[i].item()
            print(f"   Input ({x}, {y}): expected={expected}, predicted={predicted}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_attachment(model):
    """Test attaching SPD components to the model."""
    print("\n=== Testing Component Attachment ===")
    
    if model is None:
        print("❌ No model to test")
        return False
    
    try:
        # Define target modules for decomposition - using the new separate head architecture
        target_module_patterns = [
            "embed.embedding",  # nn.Embedding
            "pos_embed.pos_embedding",  # nn.Embedding
            # Attention heads - each head has separate q, k, v, o projections
            "*.q_proj",  
            "*.k_proj",
            "*.v_proj",
            "*.o_proj",  # Output projection
            # Could add more heads, but let's test with just 2 for now
            # MLP layers
            "*.W_in",  # nn.Linear
            "*.W_out",  # nn.Linear
            "unembed.linear"       # nn.Linear
        ]
        
        print(f"   Target modules: {target_module_patterns}")
        
        # Create a minimal config for ComponentModel
        class MockConfig:
            def __init__(self):
                self.C = 10
                self.gate_type = "mlp"
                self.gate_hidden_dims = [16]
                self.pretrained_model_output_attr = None
        
        config = MockConfig()
        
        # Create ComponentModel
        component_model = ComponentModel(
            base_model=model,
            target_module_patterns=target_module_patterns,
            C=config.C,
            gate_type=config.gate_type,
            gate_hidden_dims=config.gate_hidden_dims,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )
        
        print("✅ ComponentModel created successfully")
        print(f"   Number of components: {len(component_model.components)}")
        print(f"   Number of gates: {len(component_model.gates)}")
        
        # Check component names
        component_names = [k.replace("components.", "").replace("-", ".") for k in component_model.components.keys()]
        gate_names = [k.replace("gates.", "").replace("-", ".") for k in component_model.gates.keys()]
        
        print(f"   Component names: {component_names}")
        print(f"   Gate names: {gate_names}")
        
        # Test forward pass with components
        batch = torch.tensor([(0, 1, 113), (2, 3, 113)], dtype=torch.long)
        
        with torch.no_grad():
            target_out, pre_weight_acts = component_model.forward_with_pre_forward_cache_hooks(
                batch, module_names=component_names
            )
        
        print("✅ Forward pass with components successful")
        print(f"   Target output shape: {target_out.shape}")
        print(f"   Pre-weight activations keys: {list(pre_weight_acts.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component attachment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Testing Modular Addition Model Integration\n")
    
    # Test 1: Model loading
    model, config_dict = test_model_loading()
    
    # Test 2: Model inference
    inference_success = test_model_inference(model, config_dict)
    
    # Test 3: Component attachment
    component_success = test_component_attachment(model)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Model loading: {'✅' if model is not None else '❌'}")
    print(f"Model inference: {'✅' if inference_success else '❌'}")
    print(f"Component attachment: {'✅' if component_success else '❌'}")
    
    if model is not None and inference_success and component_success:
        print("\n🎉 All tests passed! Ready to run SPD experiments.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")


if __name__ == "__main__":
    main()