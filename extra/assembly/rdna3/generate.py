import os, sys, struct
sys.path.append(os.getcwd())
# PROFILE=1 to use
#os.environ["PROFILE"] = "1"
os.environ["SQTT"] = "1"
os.environ["SQTT_ITRACE_SE_MASK"] = "1"
os.environ["SQTT_LIMIT_SE"] = "1"
import xml.etree.ElementTree as ET

from tinygrad import nn, Tensor, Device
from tinygrad.helpers import get_single_element
from tinygrad.engine.realize import lower_schedule
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from extra.sqtt.attempt_sqtt_parse import parse_sqtt_print_packets

if __name__ == "__main__":
  fns = nn.state.zip_extract(Tensor.from_url("https://gpuopen.com/download/machine-readable-isa/latest/"))
  xml_str = fns['amdgpu_isa_rdna3_5.xml'].to("CPU").data()
  root = ET.fromstring(xml_str)

  a = Tensor.empty(16)+1
  for si, ei in lower_schedule(a.schedule()):
    # get text
    _, hdr, _ = elf_loader(ei.prg.lib)
    text = get_single_element([x for x in hdr if x.name==".text"]).content

    # llvm disassembler
    Device["AMD"].compiler.disassemble(ei.prg.lib)

    # run program
    ei.run()

  sqtt_events = [e for e in Device["AMD"].profile_events if isinstance(e, ProfileSQTTEvent)]
  for e in sqtt_events:
    parse_sqtt_print_packets(e.blob)

  #from hexdump import hexdump
  #hexdump(text)

  # TODO: write a disassembler
  #disassemble(text[:0x40], root)