import os

TEST_DEVICES=set([{"CPU": 0, "GPU": 1, "ANE": 2}[device] for device in map(lambda x: x.strip(), os.environ.get("TEST_DEVICES", "CPU,GPU,ANE").split(","))])
