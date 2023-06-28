#!/usr/bin/env python3

import sys
from sz import gen_stats, FileStats
from tabulate import tabulate

class RefDiff:
  def __init__(self, base: list[FileStats], pr: list[FileStats], unchanged=False):
    self.base_files, self.pr_files = base, pr

    self.base_files_set = dict({f.name: f for f in base})
    self.base_total_lc = sum(f.lines_count for f in base)

    self.pr_files_set = dict({f.name: f for f in pr})
    self.pr_total_lc = sum(f.lines_count for f in pr)

    self._modified = self.modified(unchanged)
    self._added = self.added()
    self._deleted = self.deleted()

  def format(self, files: list[dict], op: str):
    return [{**f, **{"diff": f'{f["diff"]:+}', "op": op}} for f in files]

  def files_diff_table(self):
    changes = (self.format(self._modified, "M")
               + self.format(self._added, "A")
               + self.format(self._deleted, "D"))
    return tabulate(changes, headers="keys", floatfmt=".1f", colalign=("left",) + ("right",) * 4)

  def files_line_count_diff(self, name: str):
    return self.base_files_set.get(name).lines_count - self.pr_files_set.get(name).lines_count

  def modified(self, unchanged):
    files = []
    for f in self.base_files:
      if f.name in self.pr_files_set:
        diff = self.files_line_count_diff(f.name)
        if unchanged or diff != 0:
          files.append({**f.format(), **{"diff": diff}})
    return files

  def added(self):
    files = []
    for f in self.pr_files:
      if f.name not in self.base_files_set:
        files.append({**f.format(), **{"diff": f.lines_count}})
    return files

  def deleted(self):
    files = []
    for f in self.base_files:
      if f.name not in self.pr_files_set:
        files.append({**f.format(), **{"diff": -f.lines_count}})
    return files

  def total_loc(self):
    return self.pr_total_lc

  def diff_loc(self):
    return self.pr_total_lc - self.base_total_lc


if __name__ == '__main__':
  base, pr = gen_stats(sys.argv[1]), gen_stats(sys.argv[2])
  diff = RefDiff(base, pr, unchanged=False)

  print(diff.files_diff_table(), "\n")
  print(f"total line count: {diff.total_loc()} ({diff.diff_loc():+})")

  if diff.diff_loc() < 0:
    sys.exit(1)
  sys.exit(0)
