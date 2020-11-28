#!/usr/bin/env python3

from hexdump import hexdump

# mod to make the header okay
# MH_CIGAM_64 is good
from macholib import MachO
a = MachO.MachO("model.hwx")

# load commands
for c in a.headers[0].commands:
  print(c[0])
  if c[0].cmd == 25:
    print(c[1])
    for section in c[2]:
      print(section.segname.strip(b'\0'), section.sectname.strip(b'\0'), hex(section.addr), hex(section.size), "@", hex(c[1].fileoff))
      #print(dir(section))
      if c[1].filesize > 0:
        hexdump(section.section_data)

# this parser is wrong (fixed with 64-bit one)
from macholib import SymbolTable
sym = SymbolTable.SymbolTable(a) 

syms = {}
for l in sym.nlists:
  print(l)
  if l[0].n_value != 0:
    syms[l[1]] = l[0].n_value

for k,v in syms.items():
  print(k, hex(v))



from termcolor import colored
def compare(x, y):
  ss = []
  ln = []
  ln2 = []
  for i,a in enumerate(zip(x,y)):
    if i!=0 and i%0x10 == 0:
      ss.append("%8X: " % i+''.join(ln)+"  "+''.join(ln2)+"\n")
      ln = []
      ln2 = []
    if a[0] != a[1]:
      ln.append(colored("%02X ", 'green') % a[0])
      ln2.append(colored("%02X ", 'red') % a[1])
    else:
      ln.append("%02X " % a[0])
      ln2.append("%02X " % a[1])
  return ''.join(ss)

g = MachO.MachO("model.hwx.golden")
f1 = g.headers[0].commands[1][2][0].section_data
f2 = a.headers[0].commands[1][2][0].section_data
print(compare(f1, f2))

#open("/tmp/data.section", "wb").write(f2)
#print(compare(open("model.hwx.golden", "rb").read(), open("model.hwx", "rb").read()))

