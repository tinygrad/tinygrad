import ctypes

if __name__ == "__main__":
  calc = ctypes.CDLL("./libcalculator.so")
  handle = ctypes.c_int64(-1)
  result = ctypes.c_int64(-1)
  calc.calculator_open(ctypes.create_string_buffer(b"file:///libcalculator_skel.so?calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp"),
                       ctypes.byref(handle))
  print(handle.value)
  test = (ctypes.c_int32 * 100)()
  for i in range(100): test[i] = i
  calc.calculator_sum(handle, test, 100, ctypes.byref(result))
  print(result.value)
