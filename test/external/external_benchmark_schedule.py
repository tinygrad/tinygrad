from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.helpers import Profiling, Timing, getenv
from tinygrad.engine.realize import lower_schedule

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)

  PROFILE = getenv("PROFILE", 1)
  FORWARD_ONLY = getenv("FORWARD_ONLY", 0)

  with Profiling(PROFILE):
    with Timing("***** model forward in "):
      out = mdl(img)

  if not FORWARD_ONLY:
    with Profiling(PROFILE):
      with Timing("***** model schedule in "):
        sched = out.schedule()

    # snakeviz /tmp/schedule.prof
    with Profiling(PROFILE, fn="/tmp/schedule.prof"):
      with Timing("***** model lower in "):
        eis = list(lower_schedule(sched))

  # random makes this slow
  #with Profiling(PROFILE):
  #  with Timing("***** model run in "):
  #    for ei in eis: ei.run()

  # this is all wait
  #with Profiling(PROFILE):
  #  with Timing("***** model finish in "):
  #    out.data()

