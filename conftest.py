import os, pytest, signal, threading

@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
  if os.getenv("DEV", "") == "AMD":
    os.environ["DEV"] = ":0+AMD"
  if os.getenv("DEV", "").endswith("+AMD") and getattr(config.option, "numprocesses", 0):
    config.option.numprocesses = 1

@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
  t = threading.Timer(int(os.getenv("TEST_TIMEOUT", 300)), os.kill, args=(os.getpid(), signal.SIGABRT))
  t.start()
  try: yield
  finally:
    t.cancel()
    t.join()
