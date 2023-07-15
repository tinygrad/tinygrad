from tinygrad import nn
from tinygrad.tensor import Tensor

class FPN:
  def __init__(self, in_channels_list, out_channels):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    self.top_block = LastLevelMaxPool()

  def __call__(self, x: Tensor):
    last_inner = self.inner_blocks[-1](x[-1])
    results = []
    results.append(self.layer_blocks[-1](last_inner))
    for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
    ):
      if not inner_block:
        continue
      inner_top_down = nearest_interpolate(last_inner, scale_factor=2)
      inner_lateral = inner_block(feature)
      last_inner = inner_lateral + inner_top_down
      layer_result = layer_block(last_inner)
      results.insert(0, layer_result)
    last_results = self.top_block(results[-1])
    results.extend(last_results)

    return tuple(results)

