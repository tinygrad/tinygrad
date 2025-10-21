#!/usr/bin/env python3
import os, numpy as np
from examples.openpilot.compile3 import compile as tg_compile, fetch

# --- Model ---
ONNX_URL = "https://github.com/haraschax/filedump/raw/refs/heads/master/driving_vision.onnx"
onnx_file = fetch(ONNX_URL)

# Optional: read expected logical output size from ONNX (ignores dynamic dims)
def expected_size_from_onnx(path):
    try:
        import onnx
        m = onnx.load(path)
        tot = 0
        shapes = []
        for o in m.graph.output:
            dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
            if 0 in dims or None in dims or any(d == 0 for d in dims):
                shapes.append(dims)
                continue
            sz = 1
            for d in dims: sz *= int(d)
            tot += sz
            shapes.append(dims)
        return tot, shapes
    except Exception:
        return None, None

def run_on(dev):
    os.environ["IMAGE"] = "0"
    os.environ["DEV"] = dev
    os.environ.setdefault("AMD_IFACE", "USB")
    os.environ.setdefault("NOLOCALS", "0")
    print(f"\n===== {dev} =====")
    out = tg_compile(onnx_file)
    out = np.asarray(out)
    flat = out.ravel()
    zeros = np.sum(flat == 0.0)
    print("shape:", out.shape, "dtype:", out.dtype, "len:", flat.size)
    print("zeros:", zeros, "tail(10):", flat[-10:])
    return flat

exp_total, exp_shapes = expected_size_from_onnx(onnx_file)
if exp_total is not None:
    print("Expected output scalar count (from ONNX):", exp_total, "shapes:", exp_shapes)

outs = {}
for dev in ["CL", "CPU", "AMD"]:
    try:
        outs[dev] = run_on(dev)
    except Exception as e:
        print(f"{dev} failed: {e}")
        outs[dev] = None

ref = outs.get("CL")
if ref is not None:
    for dev, arr in outs.items():
        if arr is None or dev == "CL": continue
        n = min(ref.size, arr.size)
        diff = np.abs(ref[:n] - arr[:n])
        bad = (diff > 1e-3).sum()
        print(f"\nCompare CL vs {dev}: overlap={n}, mismatches>{1e-3}: {bad}")
        if exp_total and arr.size > exp_total:
            extra = arr[exp_total:]
            print(f"{dev} has {arr.size-exp_total} extra elements. Unique(extra):", np.unique(extra))
