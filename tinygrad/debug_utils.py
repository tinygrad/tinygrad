# tinygrad/debug_utils.py
import numpy as np

def debug_realize(tensors, reference_outputs=None):
    """
    Realize a list of Tinygrad tensors in topological order and check against reference outputs.

    Args:
        tensors (list of Tensor): Tinygrad Tensor objects to realize.
        reference_outputs (dict, optional): Mapping from index to reference Tensor for comparison.
    """
    visited = set()
    order = []

    def visit(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        for inp in getattr(tensor, "inputs", []):
            visit(inp)
        order.append(tensor)

    # Visit all tensors to get topological order
    for t in tensors:
        visit(t)

    # Realize each tensor
    for i, t in enumerate(order):
        arr = t.numpy()  # force computation
        print(
            f"[{i}] Realized node: "
            f"shape={arr.shape}, dtype={arr.dtype}, device={getattr(t, 'device', 'unknown')}"
        )

        # Compare to reference if provided
        if reference_outputs is not None:
            key = list(reference_outputs.keys())[i]
            try:
                np.testing.assert_allclose(
                    arr,
                    reference_outputs[key].numpy(),
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch at node {i}: {key}"
                )
            except AssertionError:
                print(f"🚨 Output mismatch at node {i}: {key}")
                raise

    print(f"✅ Realized {len(order)} nodes successfully")
