#!/usr/bin/env python3
from difflib import unified_diff
import subprocess, os, time
from typing import List
from tinygrad.helpers import colored

subprocess.run(["git", "fetch", "origin", "master"], check=True)
ref_commit = subprocess.run(["git", "rev-parse", "tinygrad/master"], stdout=subprocess.PIPE, check=True, text=True).stdout.strip()
print(ref_commit)


"""
subprocess.run(["git", "fetch", "tinygrad", "master"], check=True)
ref_commit = subprocess.run(["git", "rev-parse", "tinygrad/master"], stdout=subprocess.PIPE, check=True, text=True).stdout.strip()
curr_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()

def replay_kernels(commit: str) -> List[str]:
  subprocess.run(['git', 'checkout', commit], check=True)
  env = os.environ.copy()
  env["DEBUG"] = "4"
  env["CI"] = "1"
  result = subprocess.run(["python3", "-m", "pytest", "test/test_ops.py", "-s"], env=env, text=True, capture_output=True)
  ret = []
  for log in result.stdout.splitlines()[6:-4]: # ignore pytest logs
    if "***" in log: continue # filter out device stats
    ret.append(log)
  return ret

if __name__ == "__main__":
  st = time.perf_counter()
  curr = replay_kernels(curr_commit)
  ref = replay_kernels(ref_commit)
  try:
    assert curr == ref
    print(colored("TESTS PASSED", "green"))
  except AssertionError:
    print(colored("REPLAY TESTS FAILED", "red"))
    diff = "\n".join(unified_diff(ref, curr, lineterm=""))
    with open(f"{curr_commit}.diff", "w") as f: f.write(diff)
  tm = (time.perf_counter() - st) * 100
  print(f"Replay tests took {tm:6.2f} ms")
"""
