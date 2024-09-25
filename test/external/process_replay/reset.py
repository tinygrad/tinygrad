#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION, os
cur = db_connection()
cur.execute(f"drop table if exists kernel_process_replay_{VERSION}")
cur.execute(f"drop table if exists schedule_diff_{VERSION}")
if os.path.exists(fp:=__file__.replace("reset", "master_schedule")):
  os.system(f"rm -rf {fp}")
