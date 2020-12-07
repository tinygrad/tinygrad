#!/usr/bin/env python3

from hexdump import hexdump

from macholib import MachO
def get_macho(fn):
  # mod to make the header okay
  # MH_CIGAM_64 is good
  dat = open(fn, "rb").read()
  dat = b"\xcf\xfa\xed\xfe"+dat[4:]
  from tempfile import NamedTemporaryFile
  with NamedTemporaryFile(delete=False) as f:
    f.write(dat)
    f.close()
  return MachO.MachO(f.name)

a = get_macho("model.hwx")

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

  ll = (max(len(x), len(y)) + 0xF)//0x10 * 0x10

  highlight = False
  next_highlight = 0x2b
  for i in range(ll+1):
    if i == next_highlight:
      highlight = True
      if i < len(y):
        next_highlight += y[i]+8
      else:
        next_highlight = None
    else:
      highlight = False
    a = "%02X" % x[i] if i < len(x) else "--", \
        "%02X" % y[i] if i < len(y) else "--"
    def fj(x):
      ss = []
      for i in range(0, 0x10, 4):
        ss.append(' '.join(x[i:i+4]))
      return '  '.join(ss)

    if i!=0 and i%0x10 == 0:
      ss.append("%8X: " % (i-0x10)+fj(ln)+"  |  "+fj(ln2)+"\n")
      ln = []
      ln2 = []
    if a[0] != a[1] and a[0] != "--" and a[1] != "--":
      ln.append(colored(a[0], 'green'))
      ln2.append(colored(a[1], 'red'))
    else:
      if highlight:
        ln.append(colored(a[0], 'yellow'))
        ln2.append(colored(a[1], 'yellow'))
      else:
        ln.append(a[0])
        ln2.append(a[1])
  return ''.join(ss)

g = get_macho("model.hwx.golden")
f1 = g.headers[0].commands[1][2][0].section_data
f2 = a.headers[0].commands[1][2][0].section_data
for i in range(0, len(f2), 0x300):
  print("===== op %d =====" % (i//0x300))
  if len(f1) < 0x300:
    print(compare(f1, f2[i:i+0x300]))
  else:
    print(compare(f1[i:i+0x300], f2[i:i+0x300]))

#open("/tmp/data.section", "wb").write(f2)
#print(compare(open("model.hwx.golden", "rb").read(), open("model.hwx", "rb").read()))

