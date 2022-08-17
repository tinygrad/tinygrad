# type: ignore
import sys

# only pyximport this
import pyximport
py_importer, pyx_importer = pyximport.install()
from accel.rawcpu.buffer import RawCPUBuffer
sys.meta_path.remove(pyx_importer)

