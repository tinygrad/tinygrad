import sys
import tabulate

def diff_sz(master_sz, branch_sz):
  diff = []
  master_files, branch_files = [set(master_sz.keys()), set(branch_sz.keys())]
  duplicate_files = master_files.intersetion(branch_files)
  for file in master_files.union(branch_files):
    if file in duplicate_files: 
      diff.append([file, master_sz[file][0] - branch_sz[file][0]])
    else:
      diff.append([file, master_sz[file][0]]) if file in master_files else diff.append([file, -branch_sz[file][0]])
      
  return diff

if __name__ == "__main__":
  headers = ["Name", "Line Diff"]
  diff = diff_sz(sys.argv[0], sys.argv[1])

  print(tabulate([headers] + sorted(diff, key=lambda x: -x[1]), headers="firstrow", floatfmt=".1f")+"\n")