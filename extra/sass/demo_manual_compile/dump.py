import argparse
from extra.sass.assembler.CubinFile import CubinFile


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile", help="cubin file")
parser.add_argument("-o", "--outfile", help="cuasm file")
args = parser.parse_args()
infile, outfile = args.infile, args.outfile
cf = CubinFile(infile)
cf.saveAsCuAsm(outfile)