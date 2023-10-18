from tinygrad.nn.state import torch_load
fn = "./weights/LLaMA/7B/consolidated.00.pth"
weights = torch_load(fn, use_new=True)
print(weights["tok_embeddings.weight"].numpy())
