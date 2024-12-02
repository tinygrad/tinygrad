from pathlib import Path
from examples.yolov8 import YOLOv8
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from tinygrad.helpers import fetch
from tinygrad.helpers import getenv
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
import numpy as np
import json

# NOTE: this will not be needed once we have f16 in WebGPU
def convert_weights_to_f32(input_file):
    with open(input_file, 'rb') as f:
        metadata_length_bytes = f.read(8)
        metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little', signed=False)
        metadata_json_bytes = f.read(metadata_length)
        float32_values = np.fromfile(f, dtype=np.float16).astype(np.float32)

    meta = json.loads(metadata_json_bytes.decode())
    for v in meta:
        if meta[v]["dtype"] == "F16":
            meta[v]["dtype"] = "F32"
            meta[v]["data_offsets"] = [meta[v]["data_offsets"][0] * 2, meta[v]["data_offsets"][1] * 2]

    new_json_bytes = json.dumps(meta).encode()
    metadata_length_bytes = len(new_json_bytes).to_bytes(8, byteorder='little', signed=False)
    f.close()
    with open(input_file, 'wb') as f:
        f.write(metadata_length_bytes)
        f.write(new_json_bytes)
        float32_values.tofile(f)


if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    yolo_variant = 'n'
    yolo_infer = YOLOv8(w=0.25, r=2.0, d=0.33, num_classes=80)
    weights_location = Path(__file__).parents[1] / "weights" / f'yolov8{yolo_variant}.safetensors'
    convert_weights_to_f32(fetch(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location))
    state_dict = safe_load(weights_location)
    load_state_dict(yolo_infer, state_dict)
    prg, inp_sizes, out_sizes, state = export_model(yolo_infer, Device.DEFAULT.lower(), Tensor.randn(1,3,640,640))
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
       text_file.write(prg)
