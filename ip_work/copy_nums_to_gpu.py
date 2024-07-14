

import numpy as np

from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import BufferCopy
from tinygrad.helpers import flat_mv


cpu_data = np.array([1, 2], dtype = np.int32)
cpu_data_memory = flat_mv(np.require(cpu_data, requirements="C").data) 
# cpu_buffer = Buffer("EXT", 2, dtypes.int)
cpu_buffer = Buffer("CPU", 2, dtypes.int)
cpu_buffer.allocate((cpu_data_memory, cpu_data))

gpu_buffer = Buffer("METAL", 2, dtypes.int)
gpu_buffer.allocate()

buffer_op = BufferCopy()
buffer_op.exec([gpu_buffer, cpu_buffer])

x = np.frombuffer(gpu_buffer.as_buffer(), dtype = np.int32)

print(x)