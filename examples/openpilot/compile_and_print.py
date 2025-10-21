from examples.openpilot.compile3 import compile as tg_compile, fetch
import numpy as np

onnx_file = fetch("https://github.com/haraschax/filedump/raw/refs/heads/master/driving_vision.onnx")

def onnx_expected_size(path):
    try:
        import onnx
        m = onnx.load(path)
        tot = 0
        for o in m.graph.output:
            dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
            if 0 in dims or None in dims or any(d==0 for d in dims): return None
            s = 1
            for d in dims: s *= int(d)
            tot += s
        return tot
    except Exception:
        return None

out_val = np.asarray(tg_compile(onnx_file))
exp = onnx_expected_size(onnx_file)
if exp is not None and out_val.size > exp:
    out_val = out_val.reshape(-1)[:exp]  # trim trailing padding
print(out_val)
print(np.sum(out_val == 0.0))
