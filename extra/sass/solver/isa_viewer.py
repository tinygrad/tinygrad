from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--key", required=False, help="Instruction key, e.g. IMAD_R_R_R_cAI")
parser.add_argument("-f", "--file", help="path to slved ASM repo file", default="/home/alvy/gbin/CuAssembler/CuAsm/InsAsmRepos/DefaultInsAsmRepos.sm_80.txt")
parser.add_argument("-l", "--list", help="list all instruction", action="store_true")

args = parser.parse_args()
ins_key, file, _list = args.key, args.file, args.list
repo = CuInsAssemblerRepos()
repo.initFromFile(file)
if _list:
  instructions = list(repo.m_InsAsmDict.keys())
  instructions.sort()
  for k in instructions: print(k)
else:
  assert ins_key
  print(repr(repo[ins_key]))
