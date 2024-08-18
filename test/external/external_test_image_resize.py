import torch
from PIL import Image
import numpy as np
from tinygrad import Tensor

pil_resize_modes = {'nearest-exact': Image.NEAREST, 'bilinear': Image.BILINEAR}
def compare_resize(image:np.ndarray, size, mode):
  torch_image = torch.tensor(image).permute(2,0,1).unsqueeze(0)
  tiny_image = Tensor(image).permute(2,0,1).unsqueeze(0)

  torch_image = torch.nn.functional.interpolate(torch_image, size, mode=mode).permute(0,2,3,1).squeeze(0).numpy()
  tiny_image = tiny_image.interpolate(size, mode).permute(0,2,3,1).squeeze(0).numpy()
  pil_image = np.array(Image.fromarray(image).resize((size[1], size[0]), pil_resize_modes[mode]))

  np.testing.assert_allclose(torch_image, pil_image)
  np.testing.assert_allclose(tiny_image, pil_image)

if __name__ == "__main__":
  for _ in range(100):
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    new_size = np.random.randint(32, 256, size=2).tolist()
    mode = np.random.choice(['nearest-exact'])
    print(mode, new_size)
    compare_resize(image, tuple(new_size), str(mode))