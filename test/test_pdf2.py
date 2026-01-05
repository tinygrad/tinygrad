import unittest
from extra.assembly.amd.pdf2 import extract, extract_tables, PDF_URLS

class TestPDF2(unittest.TestCase):
  def test_rdna3(self):
    pages = extract(PDF_URLS["rdna3"])
    assert len(pages) == 655
    assert any("RDNA3.5" in t for _, _, t in pages[2])
    assert any("SOP2 Fields" in t for _, _, t in pages[156])

  def test_rdna4(self):
    pages = extract(PDF_URLS["rdna4"])
    assert len(pages) == 711
    assert any("RDNA4" in t for _, _, t in pages[2])
    assert any("SOP2 Fields" in t for _, _, t in pages[175])

  def test_cdna(self):
    pages = extract(PDF_URLS["cdna"])
    assert len(pages) == 610
    assert any("CDNA4" in t for _, _, t in pages[2])
    assert any("SOP2 Fields" in t for _, _, t in pages[568])

  def test_rdna3_tables(self):
    pages = extract(PDF_URLS["rdna3"])
    tables = extract_tables(pages)
    assert len(tables) == 104
    assert 65 in tables and "SOP2 Fields" in tables[65][0]
    assert tables[65][1][0] == ['Field Name', 'Bits', 'Format or Description']

  def test_rdna4_tables(self):
    pages = extract(PDF_URLS["rdna4"])
    tables = extract_tables(pages)
    assert len(tables) == 115
    assert 74 in tables and "SOP2 Fields" in tables[74][0]
    assert tables[74][1][0] == ['Field Name', 'Bits', 'Format or Description']

  def test_cdna_tables(self):
    pages = extract(PDF_URLS["cdna"])
    tables = extract_tables(pages)
    assert len(tables) == 103
    assert 65 in tables and "SOP2 Fields" in tables[65][0]
    assert tables[65][1][0] == ['Field Name', 'Bits', 'Format or Description']

if __name__ == "__main__":
  unittest.main()
