import argparse
from extra.sass.assembler.CuAsmParser import CuAsmParser

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile", help="cubin file")
parser.add_argument("-o", "--outfile", help="cuasm file")
args = parser.parse_args()
infile, outfile = args.infile, args.outfile

asm_parser = CuAsmParser()
asm_parser.parse(infile)
asm_parser.saveAsCubin(outfile)