from tinygrad.helpers import tinytqdm
from tqdm import tqdm
import time

import unittest
from unittest.mock import patch
from io import StringIO
from collections import namedtuple

class TestProgressBar(unittest.TestCase):
  @patch('sys.stdout', new_callable=StringIO)
  @patch('shutil.get_terminal_size')
  def test_tqdm_bar_output(self, mock_terminal_size, mock_stdout):
    test_cases = [{'total': 100, 'ncols': 120, 'desc': 'test'},
                  {'total': 200, 'ncols': 120, 'desc': 'test'},
                  {'total': 100, 'ncols': 80, 'desc': 'test'},
                  {'total': 200, 'ncols': 80, 'desc': 'test'},
                  {'total': 100, 'ncols': 40, 'desc': ''},
                  {'total': 200, 'ncols': 40, 'desc': ''}]

    for tc in test_cases:
      mock_terminal_size.return_value = namedtuple(field_names='columns', typename='terminal_size')(tc['ncols'])

      mock_stdout.truncate(0)
      for i in tinytqdm(range(tc['total']), desc=tc['desc']):
        time.sleep(0.001)

      tinytqdm_output = mock_stdout.getvalue().split("\r")[-1].rstrip()
      iters_per_sec = float(tinytqdm_output.split("it/s")[-2].split(" ")[-1])

      tqdm_output = tqdm.format_meter(n=tc['total'], total=tc['total'], elapsed=tc['total']/iters_per_sec, ncols=tc['ncols'], prefix=tc['desc'])
      self.assertEqual(tinytqdm_output, tqdm_output)

def test_tqdm_perf():
  st = time.perf_counter()
  for i in tqdm(range(100)):
    time.sleep(0.01)

  tqdm_time = time.perf_counter() - st

  st = time.perf_counter()
  for i in tinytqdm(range(100)):
    time.sleep(0.01)
  tinytqdm_time = time.perf_counter() - st

  assert tinytqdm_time < 1.1 * tqdm_time

def test_tqdm_perf_high_iter():
  st = time.perf_counter()
  for i in tqdm(range(10^7)): pass
  tqdm_time = time.perf_counter() - st

  st = time.perf_counter()
  for i in tinytqdm(range(10^7)): pass
  tinytqdm_time = time.perf_counter() - st

  assert tinytqdm_time < 4 * tqdm_time

if __name__ == '__main__':
  unittest.main()
