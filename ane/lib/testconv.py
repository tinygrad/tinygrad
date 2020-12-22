#!/usr/bin/env python3
from ane import ANE, ANETensor

if __name__ == "__main__":
  ane = ANE()

  # 0x20 per row
  tin = ANETensor(0x60)
  tout = ANETensor(0x60)
  tw = ANETensor(0x60)

  tind = tin.data()
  toutd = tout.data()
  twd = tw.data()

  #tind[0:4] = [-1,1,-2,2]
  tind[0] =  1
  tind[0x20] = 2
  tind[0x40] = 3
  print("** before **")
  print(tind)
  print(toutd)

  # toutd[0] = \
  #   tind[0] * toutd[0] + \
  #   tind[0x20] + toutd[1] + \
  #   tind[0x40] + toutd[2]
  toutd[0] = 0x100
  toutd[1] = 0x100
  toutd[2] = 0x100

  # toutd[0x20] = \
  #   tind[0] * toutd[0x20] + \
  #   tind[0x20] + toutd[0x21] + \
  #   tind[0x40] + toutd[0x22]
  toutd[0x20] = 0x200
  toutd[0x21] = 0x200
  toutd[0x22] = 0x200

  twd[0] = 4
  twd[0x20] = 5
  twd[0x40] = 6

  """
  dat = list(open("../ops/sum.hwx", "rb").read())
  dat = bytes(dat)
  for k,v in ane.debug(dat[0x4000:0x4300], 16).items():
    print(k,v)
  comp = ane.compile(dat)
  ret = ane.run(comp, tin, tout, tw)
  """

  dat = list(open("../ops/conv.hwx", "rb").read())
  # use the output buffer as the weights
  # 6 is read 2?
  ane.fill(dat, [0x24], "B", 0x24)
  dat = bytes(dat)
  for k,v in ane.debug(dat[0x4000:0x4300], 16).items():
    print(k,v)
  comp = ane.compile(dat)
  ret = ane.run(comp, tin, tout)

  print("** after **")
  print(tind)
  print(toutd)
