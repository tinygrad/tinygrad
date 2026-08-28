import numpy as np
import unittest
from dataclasses import replace
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, Buffer
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime

class TestCommutativeParams(unittest.TestCase):
  def _test_template(self, out_slot:int, in_slot:int, variable_slots:tuple[int, ...], values:tuple[int, ...]):
    output_uop = UOp.param(out_slot, dtypes.float32, shape=(4,))
    input_uop = UOp.param(in_slot, dtypes.float32, shape=(4,))

    variables = []
    for index, slot in enumerate(variable_slots):
      variable = UOp.variable(f"var_{index}", 0, 42, dtypes.int32, param=True)
      variables.append(variable.replace(arg=replace(variable.arg, slot=slot)))

    v_range = UOp.range(4, 0)
    value = input_uop.index(v_range).load() * variables[0].cast(dtypes.float32)
    for variable in variables[1:]:
      value = value + variable.cast(dtypes.float32)
    value = output_uop.index(v_range).store(value).end(v_range)

    kernel_name = f"commutative_params_order_{out_slot}_{in_slot}_" + "_".join(str(slot) for slot in variable_slots)
    uop_graph_root = value.sink(arg=KernelInfo(name=kernel_name))
    program = to_program(uop_graph_root, Device[Device.DEFAULT].renderer)

    program_signature = program.to_elf().signature
    check_slots = [element[1] for element in program_signature]
    self.assertEqual(check_slots, list(range(len(program_signature))))

    check_tuple = (in_slot, out_slot)
    if out_slot < in_slot:
      check_tuple = (out_slot, in_slot)
    self.assertEqual(program.arg.globals, check_tuple)
    self.assertEqual(tuple(var.arg.slot for var in program.arg.vars), variable_slots)

    input_data = np.arange(4, dtype=np.float32)
    output_buf = Buffer(Device.DEFAULT, 4, dtypes.float32).allocate()
    input_buf = Buffer(Device.DEFAULT, 4, dtypes.float32, initial_value=input_data.tobytes())

    runtime = get_runtime(Device.DEFAULT, program)
    global_size, local_size = program.arg.launch_dims({})
    slot_to_buffer_map = {in_slot: input_buf, out_slot: output_buf}
    device_specific_buffers = [slot_to_buffer_map[slot]._buf for slot in program.arg.globals]
    runtime(*device_specific_buffers, global_size=global_size, local_size=local_size, vals=values, wait=True)

    expected_answer = input_data * values[0] + sum(values[1:])
    actual_answer = np.frombuffer(output_buf.as_memoryview(), dtype=np.float32)
    np.testing.assert_equal(actual_answer, expected_answer)

  def test_commutative_params(self):
    cases = [
      (0, 1, (2,), (3,)),
      (1, 0, (2,), (2,)),
      (1, 2, (0,), (3,)),
      (2, 1, (0,), (3,)),
      (0, 2, (1, 3), (4, 2)),
      (3, 2, (0, 1), (2, 4)),
    ]
    for case in cases:
      with self.subTest(case=case):
        self._test_template(*case)

  def test_params_beyond_abi_registers(self):
    # x86 passes the first 6 params in registers, the rest go on the stack
    buffer_uops = [UOp.param(i, dtypes.float32, shape=(4,)) for i in range(8)]
    v_range = UOp.range(4, 0)
    value = buffer_uops[1].index(v_range).load()
    for buffer_uop in buffer_uops[2:]:
      value = value + buffer_uop.index(v_range).load()
    value = buffer_uops[0].index(v_range).store(value).end(v_range)
    program = to_program(value.sink(arg=KernelInfo(name="params_beyond_abi_registers")), Device[Device.DEFAULT].renderer)
    self.assertEqual([element[1] for element in program.to_elf().signature], list(range(8)))

    input_data = [np.full(4, i, dtype=np.float32) for i in range(1, 8)]
    buffers = [Buffer(Device.DEFAULT, 4, dtypes.float32).allocate()]
    buffers += [Buffer(Device.DEFAULT, 4, dtypes.float32, initial_value=data.tobytes()) for data in input_data]
    global_size, local_size = program.arg.launch_dims({})
    get_runtime(Device.DEFAULT, program)(*[b._buf for b in buffers], global_size=global_size, local_size=local_size, wait=True)
    np.testing.assert_equal(np.frombuffer(buffers[0].as_memoryview(), dtype=np.float32), sum(input_data))

  @unittest.skipUnless(Device.DEFAULT == "CL", "images need CL")
  def test_image_and_flat_view_of_one_buffer(self):
    x = Tensor.rand(16, 10, 27).realize()
    weight = Tensor.rand(10).realize()
    target = Tensor.uniform(16, 27, low=0, high=10, dtype=dtypes.int32).realize()
    x_np = x.numpy()
    target_np = target.numpy()
    weight_np = weight.numpy()
    picked = np.take_along_axis(x_np, target_np[:,None,:], axis=1).squeeze(1)
    expected = -(picked * weight_np[target_np]).sum() / weight_np[target_np].sum()

    with Context(IMAGE=1):
      loss = x.nll_loss(target, weight=weight).numpy()
    np.testing.assert_allclose(loss, expected, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
  unittest.main()
