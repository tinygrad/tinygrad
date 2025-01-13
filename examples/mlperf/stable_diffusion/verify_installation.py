import torch
from diffusers import StableDiffusionPipeline

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Diffusers installed successfully!")