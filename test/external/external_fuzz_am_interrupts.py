import subprocess
import random
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def run_test(i, full_run=False):
  print(f"\rRunning iteration {i}...", end=" ", flush=True)

  p = subprocess.Popen(['python3', 'test/test_tiny.py', 'TestTiny.test_plus'],  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  if not full_run:
    time.sleep(random.uniform(0, 1200) / 1000)
    p.kill()
  else: time.sleep(3)

  try:
    stdout, stderr = p.communicate(timeout=1)
  except subprocess.TimeoutExpired:
    p.kill()
    stdout, stderr = p.communicate()

  if full_run:
    stderr_text = stderr.decode()
    print(stderr_text)
    assert "Ran 1 test in" in stderr_text and "OK" in stderr_text

max_workers = 4
with ThreadPoolExecutor(max_workers=max_workers) as executor:
  futures = []
  for i in range(1000000):
    if i % 100 == 0:
      for future in as_completed(futures):
        try: future.result()
        except Exception as e:
          print(f"\nError in iteration: {e}")
      futures = []

      run_test(i, True)
    else:
      future = executor.submit(run_test, i, False)
      futures.append(future)

    if len(futures) > max_workers * 2: futures = [f for f in futures if not f.done()]