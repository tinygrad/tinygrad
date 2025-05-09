import time, atexit, uuid
from enum import Enum

from tinygrad.device import Device
from tinygrad.helpers import ContextVar, getenv

BENCHMARK_LOG = ContextVar("BENCHMARK_LOG", "")

if BENCHMARK_LOG:
  from influxdb_client_3 import InfluxDBClient3, Point, WriteOptions, write_client_options
  from influxdb_client_3.write_client.client.write_api import WriteType

class BenchEvent(Enum):
  LOAD_WEIGHTS = "load_weights"
  STEP = "step"
  FULL = "full"
  GFLOPS = "gflops"

_events = []
def log_event_start(event:BenchEvent):
  _events.append((f"{event.value}_START", time.monotonic()))
def log_event_end(event:BenchEvent):
  _events.append((f"{event.value}_END", time.monotonic()))
def log_event_instant(event:BenchEvent, value:float|None):
  if value is None: value = time.monotonic()
  _events.append((event.value, value))

def parse_events(events:list[tuple]) -> dict[str, list[float]]:
  event_data = {}
  for event_type in BenchEvent:
    data, event_kind = [], None
    for event in events:
      if event[0] == f"{event_type.value}_START":
        start_time = event[1]
        event_kind = "interval"
      elif event[0] == f"{event_type.value}_END":
        assert event_kind == "interval"
        end_time = event[1]
        data.append((end_time - start_time))
      elif event[0] == event_type.value:
        event_kind = "single"
        data.append(event[1])
    event_data[event_type.value] = data
  return event_data

if BENCHMARK_LOG:
  INFLUXDB_HOST = getenv("INFLUXDB_HOST", "https://us-east-1-1.aws.cloud2.influxdata.com")
  INFLUXDB_ORG = getenv("INFLUXDB_ORG", "tiny")
  INFLUXDB_TOKEN = getenv("INFLUXDB_TOKEN", "")

  @atexit.register
  def write_events():
    event_data = parse_events(_events)
    if all(len(values) == 0 for values in event_data.values()):
      return

    # pull from github envvars
    ref = getenv("GITHUB_REF_NAME", "")
    commit = getenv("GITHUB_SHA", "")
    run = getenv("GITHUB_RUN_NUMBER", "")
    attempt = getenv("GITHUB_RUN_ATTEMPT", "")

    points = []
    for event_type, values in event_data.items():
      run_id = str(uuid.uuid4())
      for i, value in enumerate(values):
        point = Point(BENCHMARK_LOG.value).tag("id", run_id).tag("index", i)
        point = point.tag("device", Device.DEFAULT)
        point = point.tag("run", run).tag("attempt", attempt).tag("ref", ref).tag("commit", commit)
        point = point.field(event_type, value).field("x", run)
        points.append(point)

    write_options = WriteOptions(write_type=WriteType.synchronous, retry_interval=5000, max_retries=5, max_retry_delay=30000, exponential_base=2)
    wco = write_client_options(write_options=write_options)
    with InfluxDBClient3(host=INFLUXDB_HOST, org=INFLUXDB_ORG, token=INFLUXDB_TOKEN, database="benchmark", write_client_options=wco) as client:
      client.write(points)
