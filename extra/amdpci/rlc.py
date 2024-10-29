

# def dev.rreg(reg):
#   return pci_mmio[reg]

# def dev.wreg(reg, val):
#   pci_mmio[reg] = val

def replay_rlc(dev):
  # while dev.rreg(0x16274) != 1: pass
  # val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x80000002, hex(val)
  # val = dev.rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000091
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x80001) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000866
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x900ff) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000002
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x90000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000091
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x90001) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000866
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x90000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x866
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x200ff) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x60
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20001) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1c8
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20002) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x304
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20003) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x300ff) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x259
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30001) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3e8
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30002) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4b0
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30003) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x640
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30004) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x7d0
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30005) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x899
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30006) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8cb
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30007) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xb) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8fd
  while dev.rreg(0x16274) != 1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x30000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xc) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x8fd
  val = dev.rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x125) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x125, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000

  dev.wreg(0x3696, 0x40) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x40: pass
  #   print(dev.rreg(0x3697))
  dev.wreg(0x3696, 0x80) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x80: pass
  #   print(dev.rreg(0x3697))
  dev.wreg(0x3696, 0xc0) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0xc0: pass
  #   print(dev.rreg(0x3697))
  dev.wreg(0x3696, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x100: pass
  #   print(dev.rreg(0x3697))
  dev.wreg(0x3696, 0x140) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x140: pass
  #   print(dev.rreg(0x3697))
  val = dev.rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000
  dev.wreg(0xcc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0xce, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0xf8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0xf9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0xfa, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0xfb, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x52ef, 0x7) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1002
  dev.wreg(0x546e, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x546d, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x559f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  dev.wreg(0x559f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5464) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  dev.wreg(0x5464, 0x1340000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x546e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  dev.wreg(0x546e, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5572) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x548a, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x5489, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x599f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  dev.wreg(0x599f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5480) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  dev.wreg(0x5480, 0x1440000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x548a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  dev.wreg(0x548a, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5972) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x5452, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x5451, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x569f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  dev.wreg(0x569f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5448) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1040000
  dev.wreg(0x5448, 0x1240000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5452) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  dev.wreg(0x5452, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5672) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x541a, 0x103d1110) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x5419, 0x21c7a) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x589f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  dev.wreg(0x589f, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5410) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1040000
  dev.wreg(0x5410, 0x1040000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x541a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x103d1110
  dev.wreg(0x541a, 0x103d1010) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5872) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = dev.rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3540) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x3540, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3542) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x3542, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3544) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x3544, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3546) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x3546, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3549) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x3549, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x354b, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x354d, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354f) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1
  dev.wreg(0x354f, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x397b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  dev.wreg(0x397b, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x397c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  dev.wreg(0x397c, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x397d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  dev.wreg(0x397d, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x397e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x3
  dev.wreg(0x397e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9002) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf
  dev.wreg(0x9000, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9001, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9002, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9005, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x93d8) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  dev.wreg(0x93d8, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9017) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  dev.wreg(0x9015, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9016, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9017, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x901a, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x93dc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  dev.wreg(0x93dc, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x902c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  dev.wreg(0x902a, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x902b, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x902c, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x902f, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x93e0) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  dev.wreg(0x93e0, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9041) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xf
  dev.wreg(0x903f, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9040, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9041, 0xf) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9044, 0xf) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x93e4) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x400f
  dev.wreg(0x93e4, 0x10f) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3ab3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  dev.wreg(0x3ab3, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3ab6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3ab6, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3adc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  dev.wreg(0x3adc, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3ab4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  dev.wreg(0x3ab4, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x4185) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  dev.wreg(0x4185, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3540, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3541) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while dev.rreg(0x3541) != 0x80000000: pass # Added wait here
  val = dev.rreg(0x3541) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000, hex(val)
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  # while dev.rreg(0x5001) != 0x0: pass # Added wait here
  # val = dev.rreg(0x501b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x501b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x4f8a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x4f8a, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3b8f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  dev.wreg(0x3b8f, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3b92) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3b92, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3bb8) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  dev.wreg(0x3bb8, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3b90) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  dev.wreg(0x3b90, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x42f0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  dev.wreg(0x42f0, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3542, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3543) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while dev.rreg(0x3543) != 0x80000000: pass
  val = dev.rreg(0x3543) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x509b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x509b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x4f9a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x4f9a, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3c6b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  dev.wreg(0x3c6b, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3c6e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3c6e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3c94) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  dev.wreg(0x3c94, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3c6c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  dev.wreg(0x3c6c, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x445b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  dev.wreg(0x445b, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3544, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3545) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while dev.rreg(0x3545) != 0x80000000: pass
  val = dev.rreg(0x3545) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x511b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x511b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x4faa) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x4faa, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  val = dev.rreg(0x3d47) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xf100b
  dev.wreg(0x3d47, 0xf000b) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3d4a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3d4a, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3d70) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40
  dev.wreg(0x3d70, 0x40) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3d48) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x500000
  dev.wreg(0x3d48, 0x500000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x45c6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x70000000
  dev.wreg(0x45c6, 0x70000000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3546, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3547) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while dev.rreg(0x3547) != 0x80000000: pass
  val = dev.rreg(0x3547) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x519b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x519b, 0x2000000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x4fba) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x4fba, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5001) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5081) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5101) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x5181) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x300
  val = dev.rreg(0x64c0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x64d1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x64cf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x64d1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x64cf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x64d0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x64d0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x64d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x4f24) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3549) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3549, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x354a) != 0x80000000: pass
  val = dev.rreg(0x354a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000, hex(val)
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x651c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x652d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x652b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x652d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x652b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x652c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x652c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x6531) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x4f25) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354b) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x354b, 0x100) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x354c) != 0x80000000: pass
  val = dev.rreg(0x354c) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x6578) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x6589) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x6587) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x6589) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x6587) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x6588) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x6588) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x658d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x4f26) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354d) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x354d, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000000
  while dev.rreg(0x354e) != 0x80000000: pass
  val = dev.rreg(0x354e) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x65d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x65e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x65e3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x65e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x65e3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x2000
  val = dev.rreg(0x65e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x65e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x65e9) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x4f27) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x3555) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3555, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x354f) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x354f, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3550) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0xc0000000
  while dev.rreg(0x3550) != 0x80000000: pass
  val = dev.rreg(0x3550) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x80000000
  dev.wreg(0x3555, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1000
  dev.wreg(0x39bc, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e, hex(val) 
  val = dev.rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e
  val = dev.rreg(0x124) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x5b193e3e
  val = dev.rreg(0x16274) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x1
  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xe) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x11) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0xb0091) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x23962
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x20060) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x17ae7
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3f
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0xc0091) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x9) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x23962
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x15) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x3f
  dev.wreg(0x3696, 0x180) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5572) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = dev.rreg(0x5972) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = dev.rreg(0x5672) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  val = dev.rreg(0x5872) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3847) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3847, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x38cb) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x20070
  dev.wreg(0x38cb, 0x20070) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x38cd) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0xc0000009
  dev.wreg(0x38cd, 0xc0000009) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3846, 0x54) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3847, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e97, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e97, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e98, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e98, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e81, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e8f, 0x100) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e84, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9e7d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9e7d, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9e7c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9e7c, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9e7b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9e7b, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9e94) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9e94, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9e8e, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x1c0) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x1c0: pass

  dev.wreg(0x9ed8, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ed8, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ed9, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ed9, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ec2, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ed0, 0x100) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ec5, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9ebe) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9ebe, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9ebd) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9ebd, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9ebc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9ebc, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9ed5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9ed5, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9ecf, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x200) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x200: pass

  dev.wreg(0x9f19, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f19, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f1a, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f1a, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f03, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f11, 0x100) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f06, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9eff) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9eff, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9efe) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9efe, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9efd) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9efd, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9f16) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9f16, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f10, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x240) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x240: pass

  dev.wreg(0x9f5a, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f5a, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f5b, 0x3) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f5b, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f44, 0x1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f52, 0x100) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f47, 0x20402) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9f40) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9f40, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9f3f) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9f3f, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9f3e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9f3e, 0xffff) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x9f57) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x9f57, 0x3e80000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x9f51, 0x81010000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x280) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x280: pass

  dev.wreg(0x539e, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x134, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x13c, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x200
  dev.wreg(0x39f4, 0x200) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x52) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x39be) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39c7, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d0, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39cf, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39e1, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39ce, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d7, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39e0, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c0) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39c9, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d2, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39db, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c1) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39ca, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d3, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39dc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c2) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39cb, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d4, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39dd, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39bf) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39c8, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d1, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39da, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c3) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39cc, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39de, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x39c4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x39cd, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39d6, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x39df, 0x0) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x7fff) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x5) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x6ac000) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x6) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x2) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0x8) # amdgpu_cgs_write_register:54:(offset)
  # while dev.rreg(0x16274) != 0x1: pass 
  # TODO: ftAs

  dev.wreg(0x16274, 0x0) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x16273, 0x204e1) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x1628a, 0xa) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x16274) != 0x1: pass

  val = dev.rreg(0x16273) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x4e1
  val = dev.rreg(0x39bc) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x1000
  dev.wreg(0x39bc, 0x1000) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x397b) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = dev.rreg(0x397c) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = dev.rreg(0x397d) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = dev.rreg(0x397e) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  val = dev.rreg(0x397a) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x40202
  dev.wreg(0x3984, 0x400100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x3985) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x147faa15
  dev.wreg(0x3985, 0x147faa15) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x2c0) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x2c0: pass

  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545, hex(val)
  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545
  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545
  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545
  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545
  # val = dev.rreg(0x263e) # amdgpu_dm_plane_add_gfx11_modifiers:664:(adev->reg_offset[GC_HWIP][0][0] + 0x13de)
  # assert val == 0x545
  dev.wreg(0x3696, 0x300) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x300: pass

  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x340) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x340: pass

  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  dev.wreg(0x3696, 0x380) # amdgpu_cgs_write_register:54:(offset)
  while dev.rreg(0x3697) != 0x380: pass

  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x5db4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x44440440
  dev.wreg(0x5db4, 0x44440440) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x5db5, 0x0) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x5db6) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x22220202
  dev.wreg(0x5db6, 0x22220202) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ec) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x53ed, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x53ed, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x53ed, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53ed, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53ed) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53ed, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x53f5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x53f5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x53f5, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53f5, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53f5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53f5, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x53e5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x53e5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x53e5, 0x10100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53e5, 0x110100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53e5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x10100
  dev.wreg(0x53e5, 0x1010100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53d4) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  val = dev.rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  # assert val == 0x0
  dev.wreg(0x53d5, 0x1) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x0
  dev.wreg(0x53d5, 0x100) # amdgpu_cgs_write_register:54:(offset)
  val = dev.rreg(0x53d5) # amdgpu_cgs_read_register:47:(offset)
  assert val == 0x100
  dev.wreg(0x53d5, 0x10100) # amdgpu_cgs_write_register:54:(offset)
