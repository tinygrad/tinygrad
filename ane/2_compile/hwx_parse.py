#!/usr/bin/env python3

# mod to make the header okay
# MH_CIGAM_64 is good
from macholib import MachO
a = MachO.MachO("model.hwx")

# load commands
for c in a.headers[0].commands:
  print(c)

# this parser is wrong (fixed with 64-bit one)
from macholib import SymbolTable
sym = SymbolTable.SymbolTable(a) 

for l in sym.nlists:
  print(l)

