# challenge, implement image generation in under 150 lines. can import from tinygrad.apps.llm
from tinygrad import Tensor, nn

if __name__ == "__main__":
  vae_state_dict = nn.state.safe_load(Tensor.from_url("https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors"))
  for k,v in vae_state_dict.items():
    print(k, v.shape, v.dtype)

  state_dict = nn.state.safe_load(Tensor.from_url("https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/model.safetensors"))
  # language_model = Llama variant, apps.llm should just work
  #

  for k,v in state_dict.items():
    print(k, v.shape, v.dtype)

  # 1. Tokenise the prompt
  # 2. Run language_model.model.layers.... to embed
  # 3. Rectified flow
  # 4. VAE decode
  # 5. Save png



