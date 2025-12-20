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
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from extra.sqtt.attempt_sqtt_parse import parse_sqtt_print_packets

def disassemble(text, root:ET.Element):
  i = 0
  while i < len(text):
    ins = struct.unpack("I", text[i:i+4])[0]

    # 1. Get the encoding
    did_match = False
    for enc_el in root.findall("./ISA/Encodings/Encoding"):
      mask = enc_el.findtext("EncodingIdentifierMask")
      assert len(mask)%32 == 0
      bit_mask = int(mask, 2)
      iden = [int(x.text, 2) for x in enc_el.find("EncodingIdentifiers").findall("EncodingIdentifier")]
      for ide in iden:
        if ins&bit_mask == ide:
          did_match = True
          break
      if did_match: break
    if not did_match: raise RuntimeError(f"unknown instruction {ins:08X}")
    if len(mask) >= 64: ins = (struct.unpack("I", text[i+4:i+8])[0]<<32) | ins
    if len(mask) >= 96: ins = (struct.unpack("I", text[i+8:i+12])[0]<<64) | ins
    encoding_name = enc_el.findtext("EncodingName")

    #print(ET.tostring(enc_el).decode())

    # 2. Parse the Fields for this Encoding
    field_data = {}
    for field in enc_el.findall("MicrocodeFormat/BitMap/Field"):
      # Fields can be split into multiple ranges (RangeCount > 1)
      ranges = sorted(field.findall("BitLayout/Range"), key=lambda x: int(x.attrib.get('Order')))
      val = 0
      current_shift = 0
      for rng in ranges:
        width = int(rng.find("BitCount").text)
        chunk = (ins >> int(rng.find("BitOffset").text)) & ((1 << width) - 1)
        val |= (chunk << current_shift)
        current_shift += width
      field_data[field.find("FieldName").text] = val
    # this is already used
    del field_data["ENCODING"]

    # 3. Extract the instruction
    did_match = False
    for ins_el in root.findall("./ISA/Instructions/Instruction"):
      ins_name = ins_el.findtext("InstructionName")
      for ins_enc in ins_el.findall("InstructionEncodings/InstructionEncoding"):
        if ins_enc.findtext("EncodingName") == encoding_name:
          opcode = int(ins_enc.findtext("Opcode"))
          if "OP" in field_data and opcode == field_data["OP"]:
            did_match = True
            del field_data["OP"]
            break
        if did_match: break
      if did_match: break

    #print(ET.tostring(ins_enc).decode())
    #print()
    #print(field_data)
    if not did_match:
      print(f"{i:4X} : {ins:16x} -- {encoding_name}")
    elif did_match:
      params = []
      #print(ET.tostring(ins_el).decode())

      # 4. Extract the opcodes
      for op_ins in ins_enc.findall("Operands/Operand"):
        op_type = op_ins.findtext("OperandType")
        op_size = op_ins.findtext("OperandSize")
        op_fmt = op_ins.findtext("DataFormatName")
        op_field_name = op_ins.findtext("FieldName")
        if op_field_name is None: continue
        assert op_field_name in field_data
        # loop through operands for compare
        for op_el in root.findall("./ISA/OperandTypes/OperandType"):
          test_op_type = op_el.findtext("OperandTypeName")
          val_dict = {}
          for op_val in op_el.findall("OperandPredefinedValues/PredefinedValue"):
            val_dict[int(op_val.findtext("Value"))] = op_val.findtext("Name")
          if op_type == test_op_type:
            if field_data[op_field_name] in val_dict:
              print(op_type, op_size, op_fmt)
              params.append(val_dict[field_data[op_field_name]])
            else:
              params.append(f"{op_type}({field_data[op_field_name]})")
            del field_data[op_field_name]
            #print(op_type, op_size, op_fmt, op_el, op_field_name,
            #      field_data[op_field_name],
            #      val_dict.get(field_data[op_field_name], "<UNK>"))
            #print(ET.tostring(op_el).decode())

      print(f"{i:4X} : {ins:16x} -- {ins_name.lower()} {', '.join(params)}", field_data)

    # advance
    i += len(mask) // 8

  #print(ET.tostring(root).decode())

if __name__ == "__main__":
  # human readable manual at https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
  fns = nn.state.zip_extract(Tensor.from_url("https://gpuopen.com/download/machine-readable-isa/latest/"))
  xml_str = fns['amdgpu_isa_rdna3_5.xml'].to("CPU").data()
  with open("/tmp/rdna35.xml", "wb") as f: f.write(bytes(xml_str))
  root = ET.fromstring(xml_str)

  a = Tensor.empty(16)+1
  for ei in a.schedule():
    ei.lower()
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
