#!/usr/bin/env python3
# extract asts from process replay artifacts
import os, pickle
from test.external.process_replay.process_replay import TABLE_NAME, PAGE_SIZE, _pmap
from tinygrad.helpers import db_connection

LOGOPS = os.getenv("LOGOPS", "/tmp/sops")

def extract_ast(offset:int, name:str):
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{name}_{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset))
  for row in cur.fetchall():
    ast = str(pickle.loads(row[0])[0])
    open(LOGOPS, "a").write(str(ast).replace("\n", "").replace(" ", "")+"\n")
  return False

if __name__ == "__main__":
  _pmap("kernel", extract_ast)
