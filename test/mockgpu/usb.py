class MockUSB:
  def __init__(self, mem):
    self.mem = mem

  def read(self, address, size):
    return bytes(self.mem[address:address+size])

  def write(self, address, data):
    self.mem[address:address+len(data)] = data

  def pcie_mem_req(self, address, value=None, size=1):
    if value is None: return int.from_bytes(self.mem[address:address+size], "little")
    else: self.mem[address:address+size] = value.to_bytes(size, "little")