import csv
import sys

path = sys.argv[1]
assert "kernel_trace" in path, "needs the kernel_trace file"

with open(path, newline="") as f:
  r = csv.DictReader(f)
  for row in r:
    name = row["Kernel_Name"]
    start = int(row["Start_Timestamp"])
    end = int(row["End_Timestamp"])
    dur_ns = end - start
    dur_ms = dur_ns / 1e6

    print(f"{name}: {dur_ms:.3f} ms")
