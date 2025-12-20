from tinygrad import Tensor, nn
import xml.etree.ElementTree as ET

if __name__ == "__main__":
  # human readable manual at https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
  fns = nn.state.zip_extract(Tensor.from_url("https://gpuopen.com/download/machine-readable-isa/latest/"))
  xml_str = fns['amdgpu_isa_rdna3_5.xml'].to("CPU").data()
  root = ET.fromstring(xml_str)

  for op_el in root.findall("./ISA/OperandTypes/OperandType"):
    op_name = op_el.findtext("OperandTypeName")
    val_dict = {}
    for op_val in op_el.findall("OperandPredefinedValues/PredefinedValue"):
      val_dict[int(op_val.findtext("Value"))] = op_val.findtext("Name")
    print(op_name, val_dict)
