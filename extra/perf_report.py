from __future__ import annotations

import atexit, csv, functools, hashlib, os, subprocess, sys
from datetime import datetime

from tinygrad.helpers import CI, colored
from tinygrad.lazy import Device

"""
How to use:

$ mkdir reports  # after creating the reports directory, every run of test_speed_v_torch.py will produce a new report
$ SETBASELINE=1 python ./test/test_speed_v_torch.py  # record a baseline. compares to torch.
$ cat ./reports/latest  # check out the latest report. notice how a git commit hash and git diff are included, for reproducibility
$ cat ./reports/baseline  # check out the baseline report
$ #  make some changes to your code...
$ python ./test/test_speed_v_torch.py  # compares against baseline
$ RPT=0 python ./test/test_speed_v_torch.py  # compares against torch again (or just rm ./reports/baseline)
$ python extra/perf_report.py ./reports/baseline ./reports/latest  # compare two reports from command line
"""

@functools.lru_cache
def git_info():
  try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("Latin-1").strip()
    git_diff = subprocess.check_output(["git", "diff"]).decode("Latin-1").strip()
  except:
    print("WARNING: failed to detect git info; will not record in performance report.")
    if RPT >= 2: raise
    git_hash, git_diff = None, None
  return git_hash, git_diff

def capture_env(env_vars):
  return {k: os.getenv(k, None) for k in env_vars}

def colorize_float(x):
  ret = f"{x:7.2f}x"
  if x < 0.75:
    return colored(ret, "green")
  elif x > 1.15:
    return colored(ret, "red")
  else:
    return colored(ret, "yellow")

RPT = int(os.getenv("RPT", 1))
class PerfReport:
  required_fields = ["name", "device", "et_tinygrad", "flops", "mem"]
  stratification = ["name", "device"]
  env_to_capture = ["DEBUG", "KOPT", "OPT"]
  def __init__(self, perf_data, baseline=None, **metadata):
    self.perf_data = {PerfReport._stratum(row): row for row in perf_data}
    self.baseline = baseline
    self.metadata = metadata

  @staticmethod
  def _stratum(row): return tuple([row.get(s, None) for s in PerfReport.stratification])

  @staticmethod
  def new(baseline=None):
    git_hash, git_diff = git_info()
    return PerfReport([], baseline=baseline, **capture_env(PerfReport.env_to_capture), git_hash=git_hash, git_diff=git_diff)

  def get_fn(self):
    fn = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    if "git_hash" in self.metadata and self.metadata["git_hash"] is not None:
      fn += f"_{self.metadata['git_hash'][:7]}"
      if "git_diff" in self.metadata and self.metadata["git_diff"]: fn += f"_{hashlib.sha256(self.metadata['git_diff'].encode('Latin-1')).hexdigest()[:4]}"

    return f"{fn}.csv"

  @staticmethod
  def from_file(fn):
    with open(fn, newline="") as f: perf_data = list(csv.DictReader(f, delimiter=",", quotechar='"', escapechar="\\", quoting=csv.QUOTE_NONNUMERIC))
    # if the last row only contains metadata, that metadata applies to all rows in the file
    if perf_data[-1]["name"] == "metadata":
      perf_data, metadata = perf_data[:-1], {k: v for k, v in perf_data[-1].items() if k not in PerfReport.required_fields}
    else:
      metadata = {}
    return PerfReport(perf_data, **metadata)

  def write(self, dir="./reports/", fn=None, update_baseline=False):
    update_latest = fn is None
    if fn is None: fn = self.get_fn()
    if not self.perf_data: return  # don"t write if no perf data

    with open(os.path.join(dir, fn), "w", newline="") as f:
      fields = [*PerfReport.required_fields, *self.metadata.keys()]
      fw = csv.DictWriter(f, fields, delimiter=",", quotechar='"', escapechar="\\", quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
      fw.writeheader()
      for row in self.perf_data.values(): fw.writerow(row)
      fw.writerow({"name": "metadata", **self.metadata})

    print(f"Wrote report to {os.path.join(dir, fn)}")

    if update_latest:
      if os.path.exists(os.path.join(dir, "latest")): os.unlink(os.path.join(dir, "latest"))
      os.symlink(fn, os.path.join(dir, "latest"))
    if update_baseline:
      if os.path.exists(os.path.join(dir, "baseline")):
        print(f"Replacing old baseline: {os.readlink(os.path.join(dir, 'baseline'))}")
        os.unlink(os.path.join(dir, "baseline"))
      os.symlink(fn, os.path.join(dir, "baseline"))

  @staticmethod
  def show_row(name, et_tinygrad, flops, mem, baseline=None, baseline_flops=None, baseline_mem=None, baseline_name="torch"):
    print(("\r" if not CI else "") + f"{name:42s} "
                                     + (f"{baseline:7.2f} ms ({(baseline_flops or flops) / baseline:8.2f} GFLOPS {(baseline_mem or mem) / baseline:8.2f} GB/s) in {baseline_name}, " if baseline is not None else " " * (48 + len(baseline_name))) +
                                     f"{et_tinygrad:7.2f} ms ({flops / et_tinygrad:8.2f} GFLOPS {mem / et_tinygrad:8.2f} GB/s) in tinygrad, "
                                     + (f"{colorize_float(et_tinygrad / baseline)} {'faster' if baseline > et_tinygrad else 'slower'} " if baseline is not None else " " * 16) +
                                     f"{flops:10.2f} MOPS {mem:8.2f} MB")

  def compare(self, baseline: PerfReport):
    for row in self.perf_data.values(): self.compare_row(row, baseline)

  @staticmethod
  def compare_row(row, baseline):
    if PerfReport._stratum(row) not in baseline.perf_data:
      PerfReport.show_row(row["name"], row["et_tinygrad"], row["flops"], row["mem"], baseline_name="baseline")  # no baseline for this row
    else:
      baseline_row = baseline.perf_data[PerfReport._stratum(row)]
      PerfReport.show_row(row["name"], row["et_tinygrad"], row["flops"], row["mem"],
                          baseline=baseline_row["et_tinygrad"], baseline_flops=baseline_row["flops"], baseline_mem=baseline_row["mem"], baseline_name="baseline")

  def log_perf(self, name, et_tinygrad, flops, mem, device=Device.DEFAULT, embed_git_info=False, show=True, baseline=None):
    row = {"name": name, "device": device, "et_tinygrad": et_tinygrad, "flops": flops, "mem": mem}
    if embed_git_info: row.update(dict(zip(["git_hash", "git_diff"], git_info())))
    self.perf_data[PerfReport._stratum(row)] = row

    if show:
      if self.baseline is not None:
        PerfReport.compare_row(row, self.baseline)
      else:
        PerfReport.show_row(name, et_tinygrad, flops, mem, baseline=baseline)


if __name__ != "__main__":
  def _init_report():
    baseline_path = os.getenv("BASELINE", "./reports/baseline")
    baselinerpt = PerfReport.from_file(baseline_path) if os.path.exists(baseline_path) and RPT >= 2 else None
    rpt = PerfReport.new(baseline=baselinerpt)
    if RPT != 0 and os.path.exists("./reports"):
      atexit.register(lambda: rpt.write("./reports", update_baseline=bool(os.getenv("SETBASELINE"))))
    return rpt
  rpt = _init_report()
else:
  # compare two reports from command line
  # usage: python extra/perf_report.py <baseline> <new>
  fn1, fn2 = sys.argv[1], sys.argv[2]
  baseline, rpt = PerfReport.from_file(fn1), PerfReport.from_file(fn2)
  rpt.compare(baseline)



