
# TO RUN: python3 -m pytest -s test/test_lazy.py
# LOG DEBUG: python3 -m pytest -s -r debug --log-cli-level=DEBUG test/test_lazy.py
import numpy as np
import logging
from tinygrad import Tensor
from tinygrad.engine.lazy import create_lazybuffer, MetaOps, LazyBuffer
from tinygrad.ops import Ops
from tinygrad.codegen.kernel import split_reduceop
from tinygrad.shape.shapetracker import ShapeTracker
import pytest

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger level explicitly

def test_lazyop_two_programs():
    device = "CPU"
    shape = (2, 3)
    dtype = "float32"

    # Create tensors with data
    tensor1_data = np.random.rand(*shape).astype(np.float32)
    tensor2_data = np.random.rand(*shape).astype(np.float32)

    logger.debug(f"Tensor 1 data:\n{tensor1_data}")
    logger.debug(f"Tensor 2 data:\n{tensor2_data}")

    tensor1 = Tensor(tensor1_data)
    tensor2 = Tensor(tensor2_data)

    # Access lazy data buffers
    lazy_buffer1 = tensor1.lazydata
    lazy_buffer2 = tensor2.lazydata

    logger.debug(f"Lazy Buffer 1:\n{lazy_buffer1}")
    logger.debug(f"Lazy Buffer 2:\n{lazy_buffer2}")

    # Insert KERNEL in the middle
    kernel_buffer = split_reduceop(lazy_buffer1, Ops.SUM, (0,))

    # Reduce axis on kernel buffer
    kernel_buffer1 = kernel_buffer.reduce_axis(Ops.SUM, (0,))
    kernel_buffer2 = kernel_buffer.reduce_axis(Ops.MAX, (1,))

    # Log realized buffers
    logger.debug(f"Realized Kernel Buffer 1:\n{kernel_buffer1}")
    logger.debug(f"Realized Kernel Buffer 2:\n{kernel_buffer2}")

    assert kernel_buffer1 is not None
    assert kernel_buffer2 is not None

    # Verify ExecItem generation
    # schedule_item = ScheduleItem(kernel_buffer.ast, (kernel_buffer,))
    # exec_items = list(lower_schedule([schedule_item]))
    # assert len(exec_items) == 2  # Two ExecItems for two programs
    # assert isinstance(exec_items[0].prg, list)  # List of runners

    # Error handling: unsupported operation
    kernel_buffer.reduce_axis("UNSUPPORTED_OP", (0,))

    # Edge case: empty buffer
    empty_buffer = Tensor(np.zeros((0,))).lazydata
    with pytest.raises(ValueError):
        empty_buffer.reduce_axis(Ops.SUM, (0,))

    # Edge case: zero-sized tensor
    zero_tensor_buffer = Tensor(np.zeros((0, 0))).lazydata
    with pytest.raises(ValueError):
        zero_tensor_buffer.reduce_axis(Ops.SUM, (0,))