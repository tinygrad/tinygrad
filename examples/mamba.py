import argparse
import torch # FIXME
from tinygrad.helpers import fetch

MODELS = {
    "130m": {
        "d_model": 768,
        "n_layer": 24,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "370m": {
        "d_model": 1024,
        "n_layer": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "790m": {
        "d_model": 1536,
        "n_layer": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "1.4b": {
        "d_model": 2048,
        "n_layer": 48,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
    "2.8b": {
        "d_model": 2560,
        "n_layer": 64,
        "vocab_size": 50277,
        "pad_vocab_size_multiple": 8
    },
}

def load_model(model_name: str):
    if model_name not in MODELS.keys(): raise Exception(f"Requested unknown mamba model: {model_name}")
    downloaded = fetch(f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin?download=true")
    print(downloaded)
    weights = torch.load(downloaded)
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for LLM completion")
    parser.add_argument("--size", type=str, default="130m", help=f"Size of model to use [{', '.join([k for k in MODELS.keys()])}]")
    args = parser.parse_args()
    model = load_model(args.size)
    for layer_name in model.keys():
        print(layer_name)