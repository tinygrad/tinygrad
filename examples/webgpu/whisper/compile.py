from pathlib import Path
from examples.whisper import init_whisper
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from tinygrad.device import Device

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    model, enc = init_whisper(model_name="tiny.en")
    dummy_mel = Tensor.randn(1, 80, 3000)
    dummy_encoded_audio = model.encoder.encode(dummy_mel) # shape is model name dependent so don't want to hardcode
    dirname = Path(__file__).parent
    prg, inp_sizes, out_sizes, state = export_model(model.encoder, Device.DEFAULT.lower(), dummy_mel, model_name="whisper_encoder")
    safe_save(state, (dirname / "net_encoder.safetensors").as_posix())
    with open(dirname / f"net_encoder.js", "w") as text_file:
       text_file.write(prg)
    prg, inp_sizes, out_sizes, state = export_model(model.decoder, Device.DEFAULT.lower(), Tensor([[1, 1]]), dummy_encoded_audio, model_name="whisper_decoder")
    safe_save(state, (dirname / "net_decoder.safetensors").as_posix())
    with open(dirname / f"net_decoder.js", "w") as text_file:
       text_file.write(prg)
