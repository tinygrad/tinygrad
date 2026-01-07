#!/usr/bin/env python3
"""Test pdf.py PDF parser and enum generation."""
import unittest, tempfile, importlib.util
from extra.assembly.amd.pdf import extract, extract_tables, extract_enums, extract_pcode, write_enums, PDF_URLS

EXPECTED = {
  "rdna3": {"pages": 655, "tables": 115, "sop2_ops": 67, "sop2_first": "S_ADD_U32"},
  "rdna4": {"pages": 711, "tables": 125, "sop2_ops": 74, "sop2_first": "S_ADD_CO_U32"},
  "cdna":  {"pages": 610, "tables": 104, "sop2_ops": 52, "sop2_first": "S_ADD_U32"},
}

class TestPDF2(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.data = {name: extract(url) for name, url in PDF_URLS.items()}
    cls.tables = {name: extract_tables(pages) for name, pages in cls.data.items()}
    cls.enums = {name: extract_enums(cls.tables[name]) for name in PDF_URLS}
    cls.pcode = {name: extract_pcode(cls.data[name], cls.enums[name]) for name in PDF_URLS}

  def test_page_counts(self):
    for name, exp in EXPECTED.items():
      self.assertEqual(len(self.data[name]), exp["pages"], f"{name} page count")

  def test_table_counts(self):
    for name, exp in EXPECTED.items():
      self.assertEqual(len(self.tables[name]), exp["tables"], f"{name} table count")

  def test_tables_sequential(self):
    for name in PDF_URLS:
      nums = sorted(self.tables[name].keys())
      missing = set(range(1, max(nums) + 1)) - set(nums)
      self.assertEqual(missing, set(), f"{name} missing tables: {missing}")

  def test_generate_enums(self):
    for name, exp in EXPECTED.items():
      with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        write_enums(self.enums[name], name, f.name)
        spec = importlib.util.spec_from_file_location("enum", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Check SOP2Op
        self.assertTrue(hasattr(mod, 'SOP2Op'), f"{name} missing SOP2Op")
        self.assertEqual(len(mod.SOP2Op), exp["sop2_ops"], f"{name} SOP2Op count")
        self.assertEqual(mod.SOP2Op(0).name, exp["sop2_first"], f"{name} SOP2Op first")
        # Check all enums have at least 2 ops
        for attr in dir(mod):
          if attr.endswith('Op'):
            self.assertGreaterEqual(len(getattr(mod, attr)), 2, f"{name} {attr} has too few ops")

  def test_pcode_rdna3_tricky(self):
    """Test specific pseudocode patterns that are tricky to extract correctly."""
    pcode = self.pcode['rdna3']
    # BUFFER_ATOMIC_MAX_U64: should have 4 statements (not truncated)
    self.assertEqual(pcode[('BUFFER_ATOMIC_MAX_U64', 72)],
      'tmp = MEM[ADDR].u64;\nsrc = DATA.u64;\nMEM[ADDR].u64 = src >= tmp ? src : tmp;\nRETURN_DATA.u64 = tmp')
    # GLOBAL_STORE_B128: should have 4 MEM stores (not truncated)
    self.assertEqual(pcode[('GLOBAL_STORE_B128', 29)],
      'MEM[ADDR].b32 = VDATA[31 : 0];\nMEM[ADDR + 4U].b32 = VDATA[63 : 32];\nMEM[ADDR + 8U].b32 = VDATA[95 : 64];\nMEM[ADDR + 12U].b32 = VDATA[127 : 96]')
    # S_CMOVK_I32: should have full if/endif block
    self.assertEqual(pcode[('S_CMOVK_I32', 2)],
      "if SCC then\nD0.i32 = 32'I(signext(SIMM16.i16))\nendif")

  def test_pcode_no_examples(self):
    """Pseudocode should not contain example lines with '=>'."""
    for name in PDF_URLS:
      for (op_name, opcode), code in self.pcode[name].items():
        # No example lines (test vectors like "S_CTZ_I32_B32(0xaaaaaaaa) => 1")
        self.assertNotIn('=>', code, f"{name} {op_name} contains example line with '=>'")

if __name__ == "__main__":
  unittest.main()
