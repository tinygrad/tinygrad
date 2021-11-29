# pip3 install pyobjc-framework-MetalPerformanceShaders
from tinygrad.tensor import Function
import numpy as np
import Metal
import MetalPerformanceShaders

device = Metal.MTLCreateSystemDefaultDevice()
mtl_queue = device.newCommandQueue()
mtl_buffer = mtl_queue.commandBuffer()

class MetalBuffer:
  def __init__(self, shape, hostbuf=None):
    self.sz = np.prod(shape)*4
    if hostbuf is not None:
      self.mtl = device.newBufferWithBytes_length_options_(
        hostbuf.astype(np.float32).data,
        self.sz,
        Metal.MTLResourceStorageModeShared)
    else:
      self.mtl = device.newBufferWithLength_options_(
        self.sz,
        Metal.MTLResourceStorageModeShared)
    self.shape = shape
    self.dtype = np.float32

    self.descriptor = Metal.MTLTextureDescriptor.alloc().init()
    self.descriptor.setPixelFormat_(Metal.MTLPixelFormatR32Float)
    self.descriptor.setWidth_(np.prod(shape))

    tsz = (self.sz+15)
    tsz -= tsz%16
    self.texture = self.mtl.newTextureWithDescriptor_offset_bytesPerRow_(self.descriptor, 0, tsz)

  @staticmethod
  def fromCPU(data):
    return MetalBuffer(data.shape, data)

  def toCPU(self):
    status = mtl_buffer.status()
    #print("waiting", status)
    if status != 0:
      mtl_buffer.waitUntilCompleted()
    return np.frombuffer(b''.join(self.mtl.contents()[0:self.sz]), dtype=self.dtype).reshape(self.shape)

relu_shader = MetalPerformanceShaders.MPSImageThresholdToZero.alloc().initWithDevice_thresholdValue_linearGrayColorTransform_(device, 0, None)
add_shader = MetalPerformanceShaders.MPSImageAdd.alloc().initWithDevice_(device)

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    out = MetalBuffer(input.shape, None)
    relu_shader.encodeToCommandBuffer_sourceTexture_destinationTexture_(
      mtl_buffer, input.texture, out.texture
    )
    mtl_buffer.commit()
    return out

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors

"""
class Add(Function):
  def forward(ctx, x, y):
    #add_shader.
    pass

    #ctx.save_for_backward(x.shape, y.shape)
    #return binary_op(ctx, 'a+b', x, y)
"""

if __name__ == "__main__":
  b1 = MetalBuffer(10, np.ones(10))
  b2 = MetalBuffer(10, np.ones(10))
  out = MetalBuffer(10, None)

  add_shader.encodeToCommandBuffer_primaryTexture_secondaryTexture_destinationTexture_(
    mtl_buffer, b1.texture, b2.texture, out.texture
  )
  mtl_buffer.commit()

  print(b1.toCPU())
  print(b2.toCPU())
  print(out.toCPU())

  from tinygrad.tensor import Tensor, Device

  r1 = Tensor([-2,-1,0,1,2], device=Device.METAL)
  r2 = r1.relu()
  print(r1.cpu())
  print(r2.cpu())

