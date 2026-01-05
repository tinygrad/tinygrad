#!/usr/bin/env python3
"""Test pdf2.py PDF parser and enum generation."""
import unittest, tempfile, importlib.util
from extra.assembly.amd.pdf2 import extract, extract_tables, extract_enums, generate_enums, PDF_URLS

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
        generate_enums(extract_enums(self.tables[name]), name, f.name)
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

if __name__ == "__main__":
  unittest.main()
