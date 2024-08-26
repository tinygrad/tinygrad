#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION, getenv, os
cur = db_connection()
cur.execute(f"drop table if exists process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{getenv('GITHUB_RUN_ATTEMPT')}_{VERSION}")
cur.execute(f"drop table if exists schedule_diff_{VERSION}")
if os.path.exists(fp:=__file__.replace("reset", "master_schedule")):
  os.system(f"rm -rf {fp}")
