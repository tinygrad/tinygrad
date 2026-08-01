from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.helpers import Profiling, Timing, getenv, Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import Ops
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.uop.spec import type_verify, spec_program
import gc, os, sys

if __name__ == "__main__":
  gc.disable()
  # Every model parameter is replaced by an empty tensor below. Avoid building
  # the discarded initialization graphs in the first place.
  def empty_factory(*shape, **kwargs): return Tensor.empty(*shape, dtype=kwargs.get("dtype"))
  tensor_factories = Tensor.uniform, Tensor.ones, Tensor.zeros
  for name in ("uniform", "ones", "zeros"): setattr(Tensor, name, staticmethod(empty_factory))
  mdl = ResNet50()
  for name, factory in zip(("uniform", "ones", "zeros"), tensor_factories): setattr(Tensor, name, factory)
  img = Tensor.empty(64, 3, 224, 224)
  ren = ClangRenderer(Target(device="CPU", arch="x86_64,native"))

  PROFILE = getenv("PYPROFILE", 0)
  FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
  SCHEDULE_ONLY = getenv("SCHEDULE_ONLY", 0)
  LINEARIZE = bool(getenv("LINEARIZE", 1))

  with Timing("all "):
    with Timing("***** model tensor in    "):
      out = mdl(img)

    if not FORWARD_ONLY:
      with Timing("***** model schedule in  "):
        with Profiling(PROFILE >= 3):
          linear = out.schedule_linear()

      if not SCHEDULE_ONLY:
        asts = list({call.src[0].key:call.src[0] for call in linear.src if call.src[0].op is Ops.SINK}.values())
        if (restrict_kernel := getenv("RESTRICT_KERNEL", -1)) != -1: asts = asts[restrict_kernel:restrict_kernel+1]

        with Profiling(PROFILE, fn="/tmp/rewrite.prof"):
          with Timing("***** model rewrite in   "):
            rewritten_uops = []
            for u in asts:
              rewritten_uops.append(full_rewrite_to_sink(u, ren=ren))

        if LINEARIZE:
          with Timing("***** model linearize in "):
            uops_line = []
            for u in rewritten_uops:
              uops_line.append(linearize(u))
          with Timing("***** model verify in    "):
            for u in uops_line: type_verify(u, spec_program)
          print(sum(len(u) for u in uops_line))
  sys.stdout.flush()
  if not getenv("PGO_TRAIN"): os._exit(0)
