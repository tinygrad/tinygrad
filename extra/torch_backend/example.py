import torch, torchvision
import extra.torch_backend.backend
torch.set_default_device("tiny")

if __name__ == "__main__":
  model = torchvision.models.resnet18(pretrained=True)
  image = torch.rand(1, 3, 288, 288)
  model(image).cpu()
