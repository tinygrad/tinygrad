#!/usr/bin/env python3
# extract asts from process replay artifacts
import os, pickle
from tinygrad.helpers import db_connection, getenv, VERSION
from test.external.process_replay.process_replay import _pmap

PAGE_SIZE = 100
RUN_ID = os.getenv("GITHUB_RUN_ID", "HEAD")
TABLE_NAME = f"process_replay_{RUN_ID}_{getenv('GITHUB_RUN_ATTEMPT')}_{VERSION}"
LOGOPS = os.getenv("LOGOPS", "/tmp/sops")

def extract_ast(offset:int) -> bool:
  logops = open(LOGOPS, "a")
  conn = db_connection()
  for row in conn.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset)).fetchall():
    logops.write(str(pickle.loads(row[0])[0]).replace("\n", "").replace(" ", "")+"\n")
  return False

if __name__ == "__main__":
  conn = db_connection()
  row_count = conn.execute(f"SELECT COUNT(*) FROM '{TABLE_NAME}'").fetchone()[0]
  _pmap(row_count, extract_ast)
