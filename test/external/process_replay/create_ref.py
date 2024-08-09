# restore a specific benchmark process replay
import sqlite3, os
from tinygrad.helpers import db_connection, VERSION, getenv, tqdm

cur = db_connection()
RUN_ID = os.getenv("GITHUB_RUN_ID", "HEAD")
ATTEMPT = getenv("GITHUB_RUN_ATTEMPT")
TABLE_NAME = f"process_replay_{RUN_ID}_{ATTEMPT}_{VERSION}"

create_query = cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}';").fetchone()[0]

dest = sqlite3.connect("process_replay.db")
dest.execute(create_query)

PAGE_SIZE = 100
row_cnt = cur.execute(f"select count(*) from {TABLE_NAME}").fetchone()[0]
for offset in tqdm(range(0, row_cnt, PAGE_SIZE)):
  rows = cur.execute(f"SELECT * FROM '{TABLE_NAME}' LIMIT ? OFFSET ?", (PAGE_SIZE, offset)).fetchall()
  placeholders = ', '.join('?' * 2)
  insert_query = f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})"
  dest.executemany(insert_query, rows)
dest.execute(f"ALTER TABLE {TABLE_NAME} RENAME TO process_replay_master_{VERSION};")
