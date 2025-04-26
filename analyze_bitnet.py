from safetensors import safe_open
import os
import numpy as np 

def safetensor_analyze():
    model_dir = "/Users/viraat/Documents/projects/exla/bitnet-accelerator/bitnet_model_artifacts_b1.58-2B-4T"
    safetensors_path = os.path.join(model_dir, "model.safetensors") # Default filename for bitnet-2B

    # --- Key to Inspect ---
    # Choose a weight key that should be packed (U8)
    target_key = "model.layers.0.self_attn.q_proj.weight"
    # --------------------

    # --- Helper to decode a single U8 byte into four ternary values ---
    def decode_byte(byte_val):
        """Decodes a single U8 byte into four {-1, 0, 1} values."""
        # Ensure input is a standard Python integer if it's a numpy type
        if isinstance(byte_val, np.generic):
            byte_val = byte_val.item()
        b0 = ((byte_val       & 3) - 1) # bits 0-1
        b1 = ((byte_val >> 2) & 3) - 1  # bits 2-3
        b2 = ((byte_val >> 4) & 3) - 1  # bits 4-5
        b3 = ((byte_val >> 6) & 3) - 1  # bits 6-7
        # Return in logical order (most significant element first, like in reshape)
        return [b3, b2, b1, b0]
    # ----------------------------------------------------------------

    if not os.path.exists(safetensors_path):
        print(f"Error: File not found at {safetensors_path}")
        print("Please check the path or run the example script first to download the model.")
    else:
        print(f"Analyzing tensor '{target_key}' in: {safetensors_path}")
        try:
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                # Check if the target key exists
                if target_key not in f.keys():
                    print(f"Error: Key '{target_key}' not found in the file.")
                    print("Available keys:")
                    for k in f.keys(): print(f"  - {k}")
                else:
                    # Load the specific tensor
                    tensor = f.get_tensor(target_key)
                    print(f"\nTensor Info:")
                    print(f"  Shape: {tensor.shape}, Dtype: {tensor.dtype}")
                    

                    # Convert to numpy for easier inspection
                    tensor_np = tensor.numpy()
                    print(tensor_np)

                    # Print the first few raw U8 values (e.g., first 5 bytes of the first row)
                    print(f"\nRaw U8 values (first row, first 5 bytes):")
                    print(tensor_np[0, :5])

                    # Decode and print the corresponding ternary values
                    print(f"\nDecoded ternary values ({-1, 0, 1}) from these bytes:")
                    for i, byte_val in enumerate(tensor_np[0, :5]):
                        decoded_values = decode_byte(byte_val)
                        print(f"  Byte {i} (Value: {byte_val}): -> {decoded_values}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

    print("\nAnalysis complete.")

def gguf_analyze():
    from gguf import GGUFReader
    import os
    from pathlib import Path
    
    # Try to locate the GGUF file in common locations
    bitnet_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "bitnet"
    model_path = bitnet_dir / "ggml-model-i2_s.gguf"
    
    # Fallback paths if the primary location doesn't exist
    if not model_path.exists():
        possible_paths = [
            Path("bitnet/ggml-model-i2_s.gguf"),  # Relative to current directory
            Path("model.gguf"),  # Original hardcoded path
            Path(os.path.expanduser("~/Documents/projects/exla/tinygrad-bitnet/bitnet/ggml-model-i2_s.gguf"))
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
    
    print(f"Looking for GGUF file at: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Could not find the GGUF file at {model_path}")
        print("Please specify the correct path to the GGUF file.")
        return
        
    reader = GGUFReader(str(model_path))
    print(reader.metadata)           # View architecture info
    print(reader.tensor_infos.keys()) # List tensor names   

gguf_analyze()