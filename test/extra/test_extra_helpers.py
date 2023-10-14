#!/usr/bin/env python
import os, cloudpickle, tempfile, unittest, subprocess
from extra.helpers import enable_early_exec, cross_process, _CloudpickleFunctionWrapper

def normalize_line_endings(s): return s.replace(b'\r\n', b'\n')

class TestEarlyExec(unittest.TestCase):
  def setUp(self) -> None:
    self.early_exec = enable_early_exec()

  def early_exec_py_file(self, file_content, exec_args):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
      temp.write(file_content)
      temp_path = temp.name
    try:
      output = self.early_exec((["python3", temp_path] + exec_args, None))
      return output
    finally:
      os.remove(temp_path)

  def test_enable_early_exec(self):
    output = self.early_exec_py_file(b'print("Hello, world!")', [])
    self.assertEqual(b"Hello, world!\n", normalize_line_endings(output))

  def test_enable_early_exec_with_arg(self):
    output = self.early_exec_py_file(b'import sys\nprint("Hello, " + sys.argv[1] + "!")', ["world"])
    self.assertEqual(b"Hello, world!\n", normalize_line_endings(output))

  def test_enable_early_exec_process_exception(self):
    with self.assertRaises(subprocess.CalledProcessError):
      self.early_exec_py_file(b'raise Exception("Test exception")', [])

  def test_enable_early_exec_type_exception(self):
    with self.assertRaises(TypeError):
      self.early_exec((["python3"], "print('Hello, world!')"))

class TestCrossProcess(unittest.TestCase):

  def test_cross_process(self):
    def _iterate():
      for i in range(10): yield i
    results = list(cross_process(_iterate))
    self.assertEqual(list(range(10)), results)

  def test_cross_process_exception(self):
    def _iterate():
      for i in range(10):
        if i == 5: raise ValueError("Test exception")
        yield i
    with self.assertRaises(ValueError): list(cross_process(_iterate))

  def test_CloudpickleFunctionWrapper(self):
    def add(x, y): return x + y
    self.assertEqual(7, cloudpickle.loads(cloudpickle.dumps(_CloudpickleFunctionWrapper(add)))(3, 4))

if __name__ == '__main__':
  unittest.main()