#!/usr/bin/env python3
# extract asts from process replay artifacts
import os
from tinygrad.helpers import db_connection, VERSION
from test.external.process_replay.process_replay import _pmap

PAGE_SIZE = 100
TABLE_NAME = f"kernel_process_replay_{VERSION}"
LOGOPS = os.getenv("LOGOPS", "/tmp/sops")

def extract_ast(*args) -> bool:
  open(LOGOPS, "a").write(str(args[0]).replace("\n", "").replace(" ", "")+"\n")
  return args[-1]

if __name__ == "__main__":
  conn = db_connection()
  row_count = conn.execute(f"SELECT COUNT(*) FROM '{TABLE_NAME}'").fetchone()[0]
  _pmap("kernel", extract_ast)
