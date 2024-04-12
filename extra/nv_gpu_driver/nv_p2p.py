from hexdump import hexdump
from tinygrad.device import Device
from tinygrad.helpers import to_mv
from tinygrad.runtime.ops_nv import NVDevice, HWCopyQueue
import time

# sudo rmmod nvidia nvidia_uvm nvidia_modeset nvidia_drm
# sudo modprobe nvidia NVreg_ResmanDebugLevel=1 NVreg_RmMsg=":"
# sudo modprobe nvidia NVreg_RmMsg=":" NVreg_EnableResizableBar=1 NVreg_RegistryDwords="ForceP2P=0x11"

d0:NVDevice = Device["NV:0"]
d1:NVDevice = Device["NV:1"]
print("devices open")

b0 = d0._gpu_alloc2(0x1000)
b1 = d1._gpu_alloc2(0x1000)
print("buffers allocated")

#bc0 = to_mv(b0.base, 0x1000)
#bc1 = to_mv(b1.base, 0x1000)
#hexdump(bc0[0:0x10])
#hexdump(bc1[0:0x10])

print(b0)
print(b1)
print(f"0x{b0.base:x} 0x{b1.base:x}")

tst = memoryview(bytearray(0x1000))
tst[0] = 0xaa
tst[1] = 0xbb
d1.allocator.copyin(b1, tst)
hexdump(tst[0:0x10])

# [167448.040692] NVRM: _kgspProcessRpcEvent: received event from GPU0: 0x1005 (MMU_FAULT_QUEUED) status: 0x0 size: 32
d0._gpu_map_to_gpu(b1.base, 0x200000)

q = HWCopyQueue()
q.copy(b0.base, b1.base, 0x1000)
q.submit(d0)

tst = memoryview(bytearray(0x1000))
d0.allocator.copyout(tst, b0)
hexdump(tst[0:0x10])