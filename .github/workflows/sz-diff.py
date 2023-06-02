#!/usr/bin/env python3

import sys
from sz import gen_stats
from tabulate import tabulate
import itertools

if __name__ == '__main__':
  base, pr = gen_stats(sys.argv[1]), gen_stats(sys.argv[2])
  base_files, pr_files = {x[0]: x for x in base}, {x[0]: x for x in pr}
  base_loc, pr_loc = sum(x[1] for x in base), sum(x[1] for x in pr)
  def first_dir(x): return x.rsplit("/", 1)[0]

  modified = [[x[0], x[1], x[2], x[1]-base_files[x[0]][1], "M"] for x in pr if x[0] in base_files]
  modified = [[x[0], x[1], x[2], f'{x[3]:+}', x[4]] if x[3] != 0 else [x[0], x[1], x[2], "", ""] for x in modified]
  added = [[x[0], x[1], x[2], f'{x[1]:+}', "A"] for x in pr if x[0] not in base_files]
  deleted = [[x[0], x[1], x[2], f'{-x[1]:+}', "D"] for x in base if x[0] not in pr_files]
  files = modified+added+deleted

  dirs = []
  base_dirs_sum = {dir: sum([c[1] for c in group]) for dir, group in itertools.groupby(sorted([(first_dir(x[0]), x[1]) for x in base]), key=lambda x: x[0])}
  for dir, group in itertools.groupby(sorted([(first_dir(x[0]), x[1]) for x in pr]), key=lambda x: x[0]):
    count = sum([x[1] for x in group])
    diff = count-base_dirs_sum.get(dir, 0)
    dirs.append([dir, count, f'{diff:+}' if diff != 0 else ""])
  dirs = sorted(dirs, key=lambda x: -x[1])

  print(tabulate(files, headers=["File", "Lines", "Tokens/Line", "Diff", "Op"], floatfmt=".1f", colalign=("left", "right", "right", "right", "right"))+"\n")
  print(tabulate(dirs, headers=["Dir", "Lines", "Diff"], colalign=("left", "right", "right"))+"\n")
  print(f"total line count: {pr_loc} ({pr_loc-base_loc:+})")
