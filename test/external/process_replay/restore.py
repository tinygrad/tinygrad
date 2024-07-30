# restore a specific benchmark process replay
import pickle, os
from tinygrad.device import Device
from tinygrad.helpers import db_connection, VERSION, tqdm

cur = db_connection()
RUN_ID = os.environ["GITHUB_RUN_ID"]
ATTEMPT = os.environ["GITHUB_RUN_ATTEMPT"]
TABLE_NAME = f"process_replay_{RUN_ID}_{ATTEMPT}_{VERSION}"
PAGE_SIZE = 100
row_cnt = cur.execute(f"select count(*) from {TABLE_NAME}").fetchone()[0]
for offset in tqdm(range(0, row_cnt, PAGE_SIZE)):
  rows = cur.execute(f"SELECT val FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset)).fetchall()
  for row in rows:
    ast, opts, applied_opts, name, compare_src, ctx = pickle.loads(row[0])
    try: Device[opts.device].compiler.compile(compare_src)
    except Exception:
      print("FAILED TO COMPILE")
      print(ast)
      print(applied_opts)
      print(compare_src)
      continue
