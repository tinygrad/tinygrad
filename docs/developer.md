## Frontend

Everything in [Tensor](tensor.md) is syntactic sugar around [function.py](function.md), where the forwards and backwards passes are implemented for the different ops. That goes on to construct a graph of

::: tinygrad.lazy.LazyBuffer
    options:
        show_source: false

## Lowering

The [scheduler](/tinygrad/engine/schedule.py) converts the graph of LazyBuffers into a list of `ScheduleItem`. `ast` specifies what compute to run, and `bufs` specifies what buffers to run it on.

::: tinygrad.ops.ScheduleItem

The code in [realize](/tinygrad/engine/realize.py) lowers `ScheduleItem` to `ExecItem` with

::: tinygrad.engine.realize.lower_schedule

## Execution

Creating `ExecItem`, which has a run method

::: tinygrad.engine.realize.ExecItem
    options:
        members: true

Lists of `ExecItem` can be condensed into a single ExecItem with the Graph API (rename to Queue?)