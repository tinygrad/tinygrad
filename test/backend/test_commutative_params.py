
import numpy as np
import unittest
from tinygrad.device import Device
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad import dtypes
from tinygrad.codegen import to_program
from dataclasses import replace
from tinygrad.device import Buffer
from tinygrad.engine.realize import get_runtime

class TestCommutativeParams(unittest.TestCase):
  def _test_template(self,
           out_slot: int,
           in_slot: int,
           variable_slots: tuple[int, ...],
           values: tuple[int, ...]):
    # let's first build a micro operation
    output_uop = UOp.param(out_slot, dtypes.float32, shape=(4,))
    input_uop = UOp.param(in_slot, dtypes.float32, shape=(4,))

    variables = []

    for index, slot in enumerate(variable_slots):
      variable = UOp.variable(f"var_{index}", min_val = 0, max_val = 42,
                  dtype=dtypes.int32, param = True)
      # my idea here to create a variable to break the assumption
      # in codegen that buffers are always before variables
      replaced_arg = replace(variable.arg, slot = slot)
      variable = variable.replace(arg = replaced_arg)
      variables.append(variable)

    assert len(variables) == len(variable_slots)

    v_range = UOp.range(4, 0)

    value = input_uop.index(v_range).load()
    value = value * variables[0].cast(dtypes.float32)

    for variable in variables[1:]:
      value = value + variable.cast(dtypes.float32)

    value = output_uop.index(v_range).store(value)
    value = value.end(v_range)

    kernel_name = f"commutative_params_order_{out_slot}_{in_slot}_"
    kernel_name += "_".join(str(slot) for slot in variable_slots)

    uop_graph_root = value.sink(arg=KernelInfo(name=kernel_name))

    # self.assertTrue(Device.DEFAULT in Device)
    default_renderer = Device[Device.DEFAULT].renderer

    # okay, so now we have the program generated
    program = to_program(ast = uop_graph_root, renderer = default_renderer)

    program_signature = program.to_elf().signature

    self.assertIsNotNone(program_signature)

    # we need to make sure every parameter is in the right order
    check_slots = [element[1] for element in program_signature]
    self.assertEqual(check_slots, list(range(len(program_signature))))

    # does the pre-linearized UOp graph align? note that
    # from what I can tell the globals and vars are in from_sink
    check_tuple = (in_slot, out_slot)
    if out_slot < in_slot:
      check_tuple = (out_slot, in_slot)

    self.assertEqual(program.arg.globals, check_tuple)

    program_arg_variable_splots = tuple(var.arg.slot for var in program.arg.vars)
    self.assertEqual(program_arg_variable_splots, variable_slots)

    # now we are done with "compile time" checks time to run the kernel

    input_data = np.arange(4, dtype=np.float32)

    output_buf = Buffer(device = Device.DEFAULT,
              size = 4,
              dtype = dtypes.float32).allocate()

    # because we are passing initial_value, the allocation
    # happens automatically
    # todo(hayder): double check this later
    input_buf = Buffer(device = Device.DEFAULT,
             size = 4,
             dtype = dtypes.float32,
             initial_value = input_data.tobytes())

    runtime = get_runtime(device = Device.DEFAULT, ast = program)

    global_size, local_size = program.arg.launch_dims(var_vals = {})

    slot_to_buffer_map = {in_slot: input_buf, out_slot: output_buf}

    # put the buffers in the order of the slots
    device_specific_buffers = [slot_to_buffer_map[slot]._buf for slot in program.arg.globals]

    runtime(*device_specific_buffers,
        global_size = global_size,
        local_size = local_size,
        vals = values,
        wait = True)

    # we computed this above using the kernel
    # now doing it in plain ol' python
    expected_answer = input_data * values[0] + sum(values[1:])

    # the output should be stored in... your guessed it!... the output buffer
    actual_answer = np.frombuffer(output_buf.as_memoryview(), dtype=np.float32)

    # todo: assert_equal instead of assert_allclose *should* be the right move
    # here since we are storing an integer as a float and
    # we are not testing between multiple backends, but confirm this
    np.testing.assert_equal(actual_answer, expected_answer)

  def test_commutative_params(self):
    # now for some table driven tests (yes, i used to code in go)
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

if __name__ == "__main__":
  unittest.main()