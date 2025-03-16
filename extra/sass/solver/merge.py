
from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuInsFeeder import CuInsFeeder

import argparse
parser = argparse.ArgumentParser(usage="""
Merge multiple solved library
""")
parser.add_argument("-a", "--arch", default="sm_80")
parser.add_argument("dst", help="Destination repo path (txt)")
parser.add_argument("srcs", nargs="+", help="Source repo (txt)")

args = parser.parse_args()
arch, dst, srcs = args.arch, args.dst, args.srcs

dst_repo = CuInsAssemblerRepos(arch=arch)
for src in srcs:
  dst_repo.merge(src)

dst_repo.save2file(dst)
