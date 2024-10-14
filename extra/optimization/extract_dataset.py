#!/usr/bin/env python3
# extract asts from process replay artifacts
import os
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import db_connection
from test.external.process_replay.process_replay import _pmap
LOGOPS = os.getenv("LOGOPS", "/tmp/sops")

def extract_ast(compare, *args, **kwargs):
  if not isinstance(x:=args[0], Kernel): return str(compare)
  open(LOGOPS, "a").write(str(x.ast).replace("\n", "").replace(" ", "")+"\n")
  return compare.src

if __name__ == "__main__":
  conn = db_connection()
  _pmap(extract_ast)
