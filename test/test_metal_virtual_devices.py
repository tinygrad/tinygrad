#!/usr/bin/env python3
import os
import numpy as np
from tinygrad import Tensor, Device

def test_virtual_devices():
    # Enable metal and virtual device support
    os.environ["METAL"] = "1"
    os.environ["METAL_ENABLE_GRAPH"] = "1"
    os.environ["METAL_FORCE_SYNC"] = "1"
    
    try:
        # Create tensors on different virtual devices
        print("Creating tensors on different virtual devices...")
        
        # Create a tensor on device 0
        a = Tensor.rand(100, 100, device="METAL:0").realize()
        print(f"Created tensor a on METAL:0, shape: {a.shape}")
        
        # Move to device 1
        b = a.to("METAL:1").realize()
        print(f"Moved tensor to METAL:1, shape: {b.shape}")
        
        # Do some computation on device 1
        c = (b @ b).realize()
        print(f"Computed matrix multiplication on METAL:1, shape: {c.shape}")
        
        # Move back to device 0
        d = c.to("METAL:0").realize()
        print(f"Moved result back to METAL:0, shape: {d.shape}")
        
        # Verify results are correct by comparing with NumPy
        a_np = a.numpy()
        d_expected = a_np @ a_np
        d_actual = d.numpy()
        
        # Check if results match within tolerance
        max_diff = np.max(np.abs(d_actual - d_expected))
        print(f"Maximum difference: {max_diff}")
        print(f"Test passed: {max_diff < 1e-4}")
        
        return max_diff < 1e-4
    except Exception as e:
        print(f"Error testing virtual devices: {e}")
        return False

if __name__ == "__main__":
    success = test_virtual_devices()
    print(f"\nVirtual device test {'PASSED' if success else 'FAILED'}") 