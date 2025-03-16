from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsParser import CuInsParser

import argparse
parser = argparse.ArgumentParser(usage="""
run the parser against a cuobjdump'ed SASS file to see if parsing works.
""")
parser.add_argument("file", help="sass file path")
parser.add_argument('-a', '--arch', help="sm_80, sm_89, etc.")
args = parser.parse_args()

cnt = 0
fname, arch  = args.file, args.arch
feeder = CuInsFeeder(fname)

cip = CuInsParser(arch=arch)

for addr, code, s, ctrlcodes in feeder:
    print('0x%04x :   0x%06x   0x%028x   %s'% (addr, ctrlcodes, code, s))

    ins_key, ins_vals, ins_modi = cip.parse(s, addr, code)
    print('    Ins_Key = %s'%ins_key)
    print('    Ins_Vals = %s'%str(ins_vals))
    print('    Ins_Modi = %s'%str(ins_modi))

    cnt += 1
    if cnt>10:
        break


