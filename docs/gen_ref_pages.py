"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "tinygrad"

for path in sorted(src.rglob("*.py")):
  if not str(path).endswith("tensor.py"): continue

  module_path = path.relative_to(src).with_suffix("")
  doc_path = path.relative_to(src).with_suffix(".md")
  full_doc_path = Path("reference", doc_path)

  parts = tuple(module_path.parts)

  if parts[-1] == "__init__":
    parts = parts[:-1]
  elif parts[-1] == "__main__":
    continue

  with mkdocs_gen_files.open(full_doc_path, "w") as fd:
    identifier = ".".join(parts)
    print("::: " + identifier, file=fd)

  mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
  nav_file.writelines(nav.build_literate_nav())