from pathlib import Path
from examples.yolov8 import YOLOv8, get_weights_location
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import safe_save
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    yolo_variant = 'n'
    yolo_infer = YOLOv8(w=0.25, r=2.0, d=0.33, num_classes=80)
    state_dict = safe_load(get_weights_location(yolo_variant))
    load_state_dict(yolo_infer, state_dict)
    prg, state = TinyJit(yolo_infer).export_webgpu(Tensor.randn(1,3,640,640), tensor_names=state_dict)
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
       text_file.write(prg)
