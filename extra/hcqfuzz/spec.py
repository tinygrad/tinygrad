import os

class TestSpec:
  def prepare(self, device, seed):
    raise NotImplementedError("prepare must be implemented in the derived class")
  def get_exec_state(self):
    raise NotImplementedError("get_exec_state must be implemented in the derived class")
  def name(self): return self.__class__.__name__

class DeviceSpec:
  def prepare(self, seed):
    raise NotImplementedError("prepare must be implemented in the derived class")
  def get_exec_state(self):
    raise NotImplementedError("get_exec_state must be implemented in the derived class")
  def name(self): return self.__class__.__name__

class HCQSpec(DeviceSpec): pass
class AMDSpec(HCQSpec):
  def __init__(self):
    assert os.path.exists('/sys/module/amdgpu'), "amdgpu module should be loaded"
  
  def prepare(self, seed):
    self.env = {
      "AMD_LLVM": "0" # can randomize in the future
    }
  def get_exec_state(self): return self.env

class AMSpec(AMDSpec):
  def __init__(self):
    assert not os.path.exists('/sys/module/amdgpu'), "amdgpu module should not be loaded"

