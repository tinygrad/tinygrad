from subprocess import CalledProcessError, check_output
from tempfile import NamedTemporaryFile as Temp 
from extra.sass.assembler.utils.CubinUtils import fixCubinDesc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="path to input cuda so file")
parser.add_argument("-o", "--outfile", help="path to output sass file", required=False)
parser.add_argument("-a", "--arch", help="sm_80, sm_89, etc.", required=False)
args = parser.parse_args()
infile, outfile, arch = args.infile, args.outfile, args.arch

with Temp(mode="w+b", delete_on_close=True) as f:
  doDescHack = fixCubinDesc(infile, f.name) # , always_output=False
  if doDescHack:
    infile = f.name

  sass_b = check_output(['cuobjdump', '-arch', arch, '-sass', infile])
  if outfile:
    with open(outfile, "wb") as out:
      out.write(sass_b)
  else:
    print(sass_b.decode())
    