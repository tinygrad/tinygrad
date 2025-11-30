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

def disassemble(text, root:ET.Element):
  # TODO: write a disassembler
  i = 0
  while i < len(text):
    did_match = False
    ins = struct.unpack("I", text[i:i+4])[0]

    for enc_el in root.findall("./ISA/Encodings/Encoding"):
      if did_match: break
      name = enc_el.findtext("EncodingName")
      mask = enc_el.findtext("EncodingIdentifierMask")
      assert len(mask)%32 == 0
      bit_mask = int(mask, 2)
      iden = [int(x.text, 2) for x in enc_el.find("EncodingIdentifiers").findall("EncodingIdentifier")]
      for ide in iden:
        if ins&bit_mask == ide:
          #print("match", name, len(mask))
          #print(ET.tostring(enc_el).decode())
          did_match = True
          break
    if not did_match: raise RuntimeError(f"unknown instruction {ins:08X}")

    print(f"{i:4X} : {ins:08x} {name}")
    i += len(mask) // 8

  #print(ET.tostring(root).decode())

if __name__ == "__main__":
  # human readable manual at https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
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
  for e in sqtt_events[0:1]: # only the first SE
    parse_sqtt_print_packets(e.blob)

  disassemble(text[:0x40], root)
