import os, sys, unittest, importlib.util

# clear invalid debug var
if "DEBUG" in os.environ and not os.environ["DEBUG"].isdigit(): del os.environ["DEBUG"]

# llvm backend needs cpu device on mac
if os.getenv("LLVM") == "1" and "CPU" not in os.environ: os.environ["CPU"] = "1"

test_path = os.path.join(os.path.dirname(__file__), "speed", "external_test_speed_v_torch.py")
spec = importlib.util.spec_from_file_location("external_test_speed_v_torch", test_path)
test_module = importlib.util.module_from_spec(spec)
sys.modules["external_test_speed_v_torch"] = test_module
spec.loader.exec_module(test_module)

def patched_test_sum(self):
  def f(a, b): return a.sum()
  test_module.helper_test_generic_square('sum', 4096, f, f, onearg=True)
test_module.TestSpeed.test_sum = patched_test_sum

if __name__ == '__main__': unittest.main(module=test_module)
