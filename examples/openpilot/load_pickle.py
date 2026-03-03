import sys, pickle
from extra.bench_log import WallTimeEvent, BenchEvent

PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/openpilot.pkl"

for _ in range(10):
  with WallTimeEvent(BenchEvent.STEP): pickle.load(PKL)
