from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.helpers import Profiling, Timing, getenv
from tinygrad.ops import UOps
from tinygrad.codegen.kernel import Kernel

if __name__ == "__main__":
  mdl = ResNet50()
  img = Tensor.empty(64, 3, 224, 224)

  PROFILE = getenv("PROFILE", 1)
  FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
  SCHEDULE_ONLY = getenv("SCHEDULE_ONLY", 0)

  with Profiling(PROFILE):
    with Timing("***** model forward in "):
      out = mdl(img)

  if not FORWARD_ONLY:
    with Profiling(PROFILE):
      with Timing("***** model schedule in "):
        sched = out.schedule()

    if not SCHEDULE_ONLY:
      asts = {x.ast.key:x.ast for x in sched if x.ast.op is UOps.SINK}.values()
      kernels = []
      with Profiling(PROFILE):
        with Timing("***** model uops in "):
          for ast in asts:
            k = Kernel(ast)
            k.hand_coded_optimizations()
            kernels.append(k)

      with Profiling(PROFILE, fn="/tmp/schedule.prof"):
        with Timing("***** model linearize in "):
          for k in kernels: k.linearize()

    #renderer = Device[Device.DEFAULT].renderer
    #with Profiling(PROFILE, fn="/tmp/schedule.prof"):
    #  with Timing("***** model render in "):
    #    for n,u in uops: renderer.render(n, u)

    # snakeviz /tmp/schedule.prof
    #with Profiling(PROFILE, fn="/tmp/schedule.prof"):
    #  with Timing("***** model lower in "):
    #    eis = list(lower_schedule(sched))

  # random makes this slow
  #with Profiling(PROFILE):
  #  with Timing("***** model run in "):
  #    for ei in eis: ei.run()

  # this is all wait
  #with Profiling(PROFILE):
  #  with Timing("***** model finish in "):
  #    out.data()

