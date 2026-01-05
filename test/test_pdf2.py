import unittest
from extra.assembly.amd.pdf2 import extract, extract_tables, generate_enums, PDF_URLS

EXPECTED = {
  "rdna3": {"pages": 655, "tables": 115, "header_page": 2, "header_text": "RDNA3.5", "sop2_table": 65, "sop2_page": 156, "enums": 20, "sop2_ops": 66, "sop2_first": "S_ADD_U32"},
  "rdna4": {"pages": 711, "tables": 125, "header_page": 2, "header_text": "RDNA4", "sop2_table": 74, "sop2_page": 175, "enums": 21, "sop2_ops": 74, "sop2_first": "S_ADD_CO_U32"},
  "cdna":  {"pages": 610, "tables": 104, "header_page": 2, "header_text": "CDNA4", "sop2_table": 65, "sop2_page": 568, "enums": 18, "sop2_ops": 52, "sop2_first": "S_ADD_U32"},
}

class TestPDF2(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.data = {name: extract(url) for name, url in PDF_URLS.items()}
    cls.tables = {name: extract_tables(pages) for name, pages in cls.data.items()}
    cls.enums = {name: generate_enums(tables) for name, tables in cls.tables.items()}

  def test_page_counts(self):
    for name, exp in EXPECTED.items():
      self.assertEqual(len(self.data[name]), exp["pages"], f"{name} page count")

  def test_header_text(self):
    for name, exp in EXPECTED.items():
      texts = self.data[name][exp["header_page"]]
      self.assertTrue(any(exp["header_text"] in t for _, _, t in texts), f"{name} header")

  def test_table_counts(self):
    for name, exp in EXPECTED.items():
      self.assertEqual(len(self.tables[name]), exp["tables"], f"{name} table count")

  def test_sop2_fields(self):
    for name, exp in EXPECTED.items():
      self.assertIn(exp["sop2_table"], self.tables[name], f"{name} SOP2 table exists")
      self.assertIn("SOP2 Fields", self.tables[name][exp["sop2_table"]][0], f"{name} SOP2 title")
      self.assertEqual(self.tables[name][exp["sop2_table"]][1][0], ['Field Name', 'Bits', 'Format or Description'], f"{name} SOP2 header")
      texts = self.data[name][exp["sop2_page"]]
      self.assertTrue(any("SOP2 Fields" in t for _, _, t in texts), f"{name} SOP2 page")

  def test_enum_counts(self):
    for name, exp in EXPECTED.items():
      self.assertEqual(len(self.enums[name]), exp["enums"], f"{name} enum count")

  def test_sop2_opcodes(self):
    for name, exp in EXPECTED.items():
      self.assertIn("SOP2Op", self.enums[name], f"{name} SOP2Op exists")
      self.assertEqual(len(self.enums[name]["SOP2Op"]), exp["sop2_ops"], f"{name} SOP2Op count")
      self.assertEqual(self.enums[name]["SOP2Op"][0], exp["sop2_first"], f"{name} SOP2Op first")

if __name__ == "__main__":
  unittest.main()
