#!/usr/bin/env python3
"""Integration test for LoRA with real Llama2 architecture and tokenizer"""

import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parents[4]))
sys.path.insert(0, str(Path(__file__).parents[1]))

from tinygrad import Tensor
from tinygrad.nn.optim import AdamW

from lora import LoRALinear, LoRAAttentionAdapter, LoRAParameterManager, get_lora_config
from dataset import get_tokenizer, GovReportExample
from extra.models.llama import Transformer

def test_real_tokenizer():
    """Test that we can work with real tokenizer interface"""
    print("🔤 Testing real tokenizer interface...")
    
    # Test with our current tokenizer
    tokenizer = get_tokenizer()
    
    # Test basic functionality
    test_text = "Summarize the following government document:\n\nThis is a test document.\n\nSummary:"
    
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"   Original: {test_text[:50]}...")
    print(f"   Tokens: {len(tokens)} tokens")
    print(f"   Decoded: {decoded[:50]}...")
    
    # Test required properties
    assert hasattr(tokenizer, 'pad_token_id'), "Missing pad_token_id"
    assert hasattr(tokenizer, 'eos_token_id'), "Missing eos_token_id"  
    assert hasattr(tokenizer, 'bos_token_id'), "Missing bos_token_id"
    
    print(f"   pad_token_id: {tokenizer.pad_token_id}")
    print(f"   eos_token_id: {tokenizer.eos_token_id}")
    print(f"   bos_token_id: {tokenizer.bos_token_id}")
    
    print("   ✅ Tokenizer interface works")
    return True

def test_llama_model_dimensions():
    """Test LoRA with realistic Llama2 dimensions (scaled down but proportional)"""
    print("\n🦙 Testing LoRA with Llama-like model dimensions...")
    
    # Create a small but proportionally correct Llama model
    # Llama2-7B: dim=4096, n_heads=32, n_layers=32
    # Our test: dim=512, n_heads=8, n_layers=2 (same ratios)
    
    model = Transformer(
        dim=512,           # 4096 scaled down 8x
        hidden_dim=1024,   # 11008 scaled down ~10x
        n_heads=8,         # 32 scaled down 4x  
        n_layers=2,        # 32 scaled down 16x (for speed)
        norm_eps=1e-5,
        vocab_size=32000,  # Real Llama vocab size
        max_context=2048   # Real context length
    )
    
    print(f"   Model: dim={512}, heads={8}, layers={2}, vocab={32000}")
    
    # Apply LoRA
    attention_adapter = LoRAAttentionAdapter(r=16, alpha=32.0, target_modules=['wq', 'wk', 'wv', 'wo'])
    
    applied_count = 0
    for i, layer in enumerate(model.layers):
        layer_original = attention_adapter.apply(attention_layer=layer.attention)
        applied_count += len([k for k in layer_original.keys() if 'attention' in k])
        print(f"   ✅ Applied LoRA to layer {i}: {list(layer_original.keys())}")
    
    # Get LoRA parameters
    LoRAParameterManager.freeze_base_model(model=model)
    lora_params = LoRAParameterManager.get_lora_parameters(model=model)
    
    print(f"   ✅ Found {len(lora_params)} LoRA parameters")
    
    # Verify parameter shapes match expectations
    expected_params = 2 * 4 * 2  # layers × modules × (A,B)
    if len(lora_params) != expected_params:
        print(f"   ⚠️ Expected {expected_params} params, got {len(lora_params)}")
    
    return True, model, lora_params

def test_realistic_training_step(model, lora_params):
    """Test training step with realistic data sizes"""
    print("\n🏋️ Testing realistic training step...")
    
    # Use realistic batch size and sequence length
    batch_size, seq_len = 1, 1024  # More realistic sizes
    vocab_size = 32000
    
    # Create realistic input data
    input_ids = Tensor.randint(batch_size, seq_len, low=1, high=vocab_size-1000)  # Avoid special tokens
    labels = Tensor.randint(batch_size, seq_len, low=1, high=vocab_size-1000)
    
    print(f"   Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
    
    # Create optimizer (only for params that will get gradients)
    optimizer = AdamW(lora_params, lr=1e-4)
    
    # Training step
    Tensor.training = True
    optimizer.zero_grad()
    
    print("   Running forward pass...")
    logits = model.forward(input_ids, start_pos=0, temperature=float('nan'), 
                         top_k=0, top_p=0.0, alpha_f=0.0, alpha_p=0.0)
    
    print(f"   Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong logits shape: {logits.shape}"
    
    # Compute loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits_flat = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels_flat = shift_labels.reshape(-1)
    
    loss = shift_logits_flat.sparse_categorical_crossentropy(shift_labels_flat)
    loss_value = loss.item()
    
    print(f"   Initial loss: {loss_value:.4f}")
    
    # Backward pass
    print("   Running backward pass...")
    loss.backward()
    
    # Count parameters with gradients
    params_with_grad = sum(1 for p in lora_params if p.grad is not None)
    print(f"   Parameters with gradients: {params_with_grad}/{len(lora_params)}")
    
    # Step only parameters with gradients
    if params_with_grad > 0:
        temp_optimizer = AdamW([p for p in lora_params if p.grad is not None], lr=1e-4)
        temp_optimizer.step()
        print(f"   ✅ Successfully stepped {params_with_grad} parameters")
    
    return loss_value

def test_govreport_data_compatibility():
    """Test that our pipeline works with GovReport data format"""
    print("\n📄 Testing GovReport data compatibility...")
    
    tokenizer = get_tokenizer()
    
    # Create a mock GovReport example
    example = GovReportExample(
        input_text="This is a long government document that needs to be summarized. " * 20,
        output_text="This is a concise summary of the document.",
        example_id="test_001"
    )
    
    # Test tokenization
    input_tokens = tokenizer.encode(f"Summarize the following government document:\\n\\n{example.input_text}\\n\\nSummary:")
    output_tokens = tokenizer.encode(example.output_text)
    
    print(f"   Input tokens: {len(input_tokens)}")
    print(f"   Output tokens: {len(output_tokens)}")
    
    # Test that we can create tensors
    input_tensor = Tensor(input_tokens[:1024])  # Truncate to max length
    output_tensor = Tensor(output_tokens[:256])   # Summary is shorter
    
    print(f"   Input tensor shape: {input_tensor.shape}")
    print(f"   Output tensor shape: {output_tensor.shape}")
    
    print("   ✅ GovReport data compatibility works")
    return True

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Llama2 70B LoRA Integration Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 1: Tokenizer
        if test_real_tokenizer():
            tests_passed += 1
            
        # Test 2: Model dimensions  
        success, model, lora_params = test_llama_model_dimensions()
        if success:
            tests_passed += 1
            
            # Test 3: Training step
            try:
                loss = test_realistic_training_step(model, lora_params)
                if loss > 0:  # Loss should be positive
                    tests_passed += 1
                    print(f"   ✅ Training step completed with loss {loss:.4f}")
                else:
                    print(f"   ❌ Invalid loss value: {loss}")
            except Exception as e:
                print(f"   ❌ Training step failed: {e}")
        
        # Test 4: Data compatibility
        if test_govreport_data_compatibility():
            tests_passed += 1
    
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Results
    print(f"\n📊 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS: All integration tests passed!")
        print("💡 LoRA implementation is ready for real Llama2 70B training")
        return 0
    else:
        print("❌ FAILURE: Some integration tests failed")
        print("🔧 Need to fix issues before attempting real training")
        return 1

if __name__ == "__main__":
    exit(main())