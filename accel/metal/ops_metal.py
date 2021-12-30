# pip3 install pyobjc-framework-MetalPerformanceShaders
from tinygrad.tensor import Function
from tinygrad.helpers import binary_broadcast
import numpy as np
import Metal
import MetalPerformanceShaders

device = Metal.MTLCreateSystemDefaultDevice()
mtl_queue = device.newCommandQueue()
mtl_buffers = []

def cmd_buffer():
  ret = mtl_queue.commandBuffer()
  mtl_buffers.append(ret)
  return ret

class MetalBuffer:
  def __init__(self, shape, hostbuf=None):
    self.sz = np.prod(shape)*4
    # TODO: fix this limit
    assert self.sz < 16384
    if hostbuf is not None:
      if isinstance(hostbuf, MetalBuffer):
        self.mtl = hostbuf.mtl
      else:
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
    global mtl_buffers
    for b in mtl_buffers:
      b.waitUntilCompleted()
    mtl_buffers = []
    return np.frombuffer(b''.join(self.mtl.contents()[0:self.sz]), dtype=self.dtype).reshape(self.shape)

relu_shader = MetalPerformanceShaders.MPSImageThresholdToZero.alloc().initWithDevice_thresholdValue_linearGrayColorTransform_(device, 0, None)
inv_relu_shader = MetalPerformanceShaders.MPSImageThresholdBinary.alloc().initWithDevice_thresholdValue_maximumValue_linearGrayColorTransform_(device, 0, 1, None)
add_shader = MetalPerformanceShaders.MPSImageAdd.alloc().initWithDevice_(device)
sub_shader = MetalPerformanceShaders.MPSImageSubtract.alloc().initWithDevice_(device)
mul_shader = MetalPerformanceShaders.MPSImageMultiply.alloc().initWithDevice_(device)
sum_shader = MetalPerformanceShaders.MPSImageReduceRowSum.alloc().initWithDevice_(device)

def unary_op(shader, input):
  out = MetalBuffer(input.shape, None)
  mtl_buffer = cmd_buffer()
  shader.encodeToCommandBuffer_sourceTexture_destinationTexture_(
    mtl_buffer, input.texture, out.texture
  )
  mtl_buffer.commit()
  return out

def binary_op(shader, x, y):
  ret = MetalBuffer(x.shape, None)
  mtl_buffer = cmd_buffer()
  shader.setPrimaryEdgeMode_(MetalPerformanceShaders.MPSImageEdgeModeClamp)
  shader.setSecondaryEdgeMode_(MetalPerformanceShaders.MPSImageEdgeModeClamp)
  shader.encodeToCommandBuffer_primaryTexture_secondaryTexture_destinationTexture_(
    mtl_buffer, x.texture, y.texture, ret.texture
  )
  mtl_buffer.commit()
  return ret

class Sum(Function):
  def forward(ctx, input, axis=None):
    assert axis is None or len(axis) == len(input.shape)
    ctx.save_for_backward(input.shape, axis)
    out = MetalBuffer((1,), None)
    mtl_buffer = cmd_buffer()
    sum_shader.encodeToCommandBuffer_sourceTexture_destinationTexture_(
      mtl_buffer, input.texture, out.texture
    )
    mtl_buffer.commit()
    return out

  def backward(ctx, grad_output):
    shape, axis = ctx.saved_tensors
    out = MetalBuffer(shape, None)
    return binary_op(add_shader, out, grad_output)

class ReLU(Function):
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return unary_op(relu_shader, input)

  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return binary_op(mul_shader, unary_op(inv_relu_shader, input), grad_output)

class Add(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(add_shader, x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return grad_output, grad_output

class Sub(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(sub_shader, x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    out = MetalBuffer(y.shape, None)
    return grad_output, binary_op(sub_shader, out, grad_output)

class Mul(Function):
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return binary_op(mul_shader, x, y)

  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    grad_x = binary_op(mul_shader, y, grad_output)
    grad_y = binary_op(mul_shader, x, grad_output)
    return grad_x, grad_y

class Reshape(Function):
  def forward(ctx, x, shape):
    ctx.save_for_backward(x.shape)
    # TODO: move this into global reshape?
    shape = tuple(-np.prod(x.shape) // np.prod(shape) if s == -1 else s for s in shape)
    return MetalBuffer(shape, x)

  def backward(ctx, grad_output):
    in_shape, = ctx.saved_tensors
    return MetalBuffer(in_shape, grad_output)

# METAL=1 python3 test/test_ops.py TestOps.test_relu
if __name__ == "__main__":
  b1 = MetalBuffer(10, np.ones(10))
  b2 = MetalBuffer(10, np.ones(10))
  out = MetalBuffer(10, None)

  mtl_buffer = cmd_buffer()
  add_shader.encodeToCommandBuffer_primaryTexture_secondaryTexture_destinationTexture_(
    mtl_buffer, b1.texture, b2.texture, out.texture
  )
  mtl_buffer.commit()

  print(b1.toCPU())
  print(b2.toCPU())
  print(out.toCPU())

  from tinygrad.tensor import Tensor, Device

  r1 = Tensor([-2,-1,0,2,4], device=Device.METAL)
  r2 = r1.relu()
  r3 = r2.sum()
  r3.backward()
  print(r1.cpu())
  print(r2.cpu())
  print(r3.cpu())

