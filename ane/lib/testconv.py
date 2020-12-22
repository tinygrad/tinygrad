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
  tind[0x20] = -2
  tind[0x40] = 3

  # toutd[0] = \
  #   tind[0] * twd[0] + \
  #   tind[0x20] + twd[1] + \
  #   tind[0x40] + twd[2]

  twd[0] = 4
  twd[1] = 0x100

  twd[0x20] = 5
  twd[0x21] = 5
  twd[0x22] = 5

  twd[0x40] = 6

  print("** before **")
  print(tind)
  print(toutd)

  """
  dat = list(open("../ops/sum.hwx", "rb").read())
  dat = bytes(dat)
  for k,v in ane.debug(dat[0x4000:0x4300], 16).items():
    print(k,v)
  comp = ane.compile(dat)
  ret = ane.run(comp, tin, tout, tw)
  """

  datb = open("../ops/sum.hwx", "rb").read()
  dat = open("../ops/conv.hwx", "rb").read()
  dd = ane.unpack(dat[0x4000:0x4300])
  # use the 3rd arg as the weights
  dd["aneTD.Header[9].KBase0"] = 6
  dd["aneRegs.NE.PostScale.PostScale"] = 0x3c00
  #dd["aneRegs.L2.L2Cfg.InputReLU"] = 1
  #dd["aneRegs.NE.MACCfg.NonlinearMode"] = 1
  #dd["aneRegs.TileDMADst.Fmt.MemFmt"] = 0
  #dd["aneRegs.L2.ResultBase.Addr"] = 0
  #dd["aneRegs.Common.ChCfg.InFmt"] = 1
  #dd["aneRegs.TileDMADst.Fmt.ZeroPadFirst"] = 0
  #dd["aneRegs.TileDMADst.DMAConfig.En"] = 0
  for k,v in dd.items():
    print(k,v)
  dat = datb[:0x4000] + ane.pack(dd, dat[0x4000:0x4300]) + datb[0x4300:]
  comp = ane.compile(dat)
  ret = ane.run(comp, tin, tout, tw)

  print("** after **")
  print(tind)
  print(toutd)
