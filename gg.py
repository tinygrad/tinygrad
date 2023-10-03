from pathlib import Path
from extra.gguf import gguf_load

gguf_file_path = Path(__file__).parents[0] / "weights/llama2-7b-q4/llama-2-7b.Q4_0.gguf"
gguf_load(str(gguf_file_path))
