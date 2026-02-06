# resnet_debug_repro.py
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import numpy as np
import tempfile
import urllib.request
from tinygrad.tensor import Tensor
from tinygrad.debug_utils import debug_realize  # Ensure debug_utils.py is in tinygrad/

# ----------------------------
# Step 1: Download ONNX model
# ----------------------------
model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx"
tmp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx").name
urllib.request.urlretrieve(model_url, tmp_model_path)

# ----------------------------
# Step 2: Load model and infer shapes
# ----------------------------
onnx_model = onnx.load(tmp_model_path)
inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
model = gs.import_onnx(inferred_model)

# ----------------------------
# Step 3: Rewrite model to output all intermediate nodes
# ----------------------------
all_node_outputs = [o for n in model.nodes for o in n.outputs]
model.outputs = all_node_outputs

# Save rewritten model to temporary file
rewritten_model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx").name
onnx.save(gs.export_onnx(model), rewritten_model_path)

# ----------------------------
# Step 4: Run ONNX Runtime to get all outputs
# ----------------------------
sess = ort.InferenceSession(rewritten_model_path)

# Generate random inputs for dynamic dimensions
inputs = {}
for inp in sess.get_inputs():
    shape = [1 if (dim is None or isinstance(dim, str)) else dim for dim in inp.shape]
    inputs[inp.name] = np.random.rand(*shape).astype(np.float32)

# Run the model
onnx_outputs = sess.run(None, inputs)

# ----------------------------
# Step 5: Wrap outputs in Tinygrad Tensors
# ----------------------------
tiny_tensors = []
for i, out in enumerate(onnx_outputs):
    t = Tensor(out)
    tiny_tensors.append(t)
    print(f"[{i}] Wrapped node: shape={t.shape}, dtype={t.dtype}, device={t.device}")

# ----------------------------
# Step 6: Debug realize
# ----------------------------
debug_realize(
    tiny_tensors,
    reference_outputs={i: t for i, t in enumerate(tiny_tensors)}
)

print("✅ Full debug run completed")
