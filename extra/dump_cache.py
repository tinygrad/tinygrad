import sys
import sqlite3

if __name__ == "__main__":
  fn = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinygrad_cache"
  conn = sqlite3.connect(fn)
  cur = conn.cursor()
  cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
  for f in cur.fetchall():
    table = f[0]
    cur2 = conn.cursor()
    cur2.execute(f"SELECT COUNT(*) FROM {table}")
    cnt = cur2.fetchone()[0]
    print(f"{table:20s} : {cnt}")
