from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.helpers import Profiling, Timing, getenv
from tinygrad.engine.realize import lower_schedule

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)

  PROFILE = getenv("PROFILE", 1)

  with Profiling(PROFILE):
    with Timing("***** model forward in "):
      out = mdl(img)

  with Profiling(PROFILE):
    with Timing("***** model schedule in "):
      sched = out.schedule()

  with Profiling(PROFILE):
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

