from pathlib import Path
from examples.yolov8 import YOLOv8
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from extra.utils import download_file
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load, load_state_dict

if __name__ == "__main__":
    yolo_variant = 'n'
    yolo_infer = YOLOv8(w=0.25, r=2.0, d=0.33, num_classes=80)

    weights_location = Path(__file__).parents[1] / "weights" / f'yolov8{yolo_variant}.safetensors'
    download_file(f'https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8{yolo_variant}.safetensors', weights_location)

    state_dict = safe_load(weights_location)
    load_state_dict(yolo_infer, state_dict)

    mode = "webgl" if getenv("WEBGL", "") != "" else ""
    prg, inp_sizes, out_sizes, state = export_model(yolo_infer, mode, Tensor.randn(1,3,640,640))
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
       text_file.write(prg)
