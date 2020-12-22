#!/usr/bin/env python3
from ane import ANE, ANETensor

if __name__ == "__main__":
  ane = ANE()

  # 0x20 per row
  tin = ANETensor(0x60)
  tout = ANETensor(0x60)

  tind = tin.data()
  toutd = tout.data()

  #tind[0:4] = [-1,1,-2,2]
  tind[0] =  1
  tind[0x20] = 2
  tind[0x40] = 3
  print("** before **")
  print(tind)
  print(toutd)

  comp = ane.compile(open("../ops/conv.hwx", "rb").read())
  ret = ane.run(comp, tin, tout)
  print("** after **")
  print(tind)
  print(toutd)
