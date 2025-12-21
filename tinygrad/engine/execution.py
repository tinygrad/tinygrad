from __future__ import annotations
from typing import Any
from tinygrad.helpers import DEBUG, GlobalCounters, all_same, dedup, colored, ansilen, PROFILE, ProfilePointEvent, cpu_events, time_to_str, TRACEMETA
from tinygrad.uop.ops import UOp, Ops, sym_infer
from tinygrad.device import Device, Buffer

# **************** ExecutionUnit ****************

class ExecutionUnit:
  """
  A bound, ready-to-execute unit. Replaces CapturedJit.

  Takes ExecItems and binds them to real device resources on execution.
  """

  def __init__(self, items: list):
    """
    Create an ExecutionUnit from ExecItems.

    Args:
      items: ExecItems (with bufs as Buffers or UOps, lib optionally set)
    """
    from tinygrad.engine.realize import ExecItem

    self.items: list[ExecItem] = items
    self.buffer_map: dict[UOp, Buffer] = {}
    self.var_vals: dict[str, int] = {}

    # Create bound items with runners - lazy, done on first call
    self._bound_items: list[tuple[Any, list[Buffer], tuple, dict[str, int]]]|None = None
    self._graphs: list|None = None
    self._first_run = True

  def _bind(self):
    """Create runners from lib and bind buffers."""
    from tinygrad.engine.realize import CompiledRunner, BufferCopy, BufferXfer, ViewOp, EncDec, get_runner, get_program

    self._bound_items = []
    for item in self.items:
      # Get buffers - either from buffer_map (for UOps) or directly (for already-bound Buffers)
      bufs: list[Buffer] = []
      for b in item.bufs:
        if b is None:
          continue
        if isinstance(b, UOp):
          bufs.append(self.buffer_map[b])
        else:
          bufs.append(b)

      # Create runner from lib or use existing prg
      if item.prg is not None:
        runner = item.prg
      elif item.ast.op is Ops.SINK:
        device = bufs[0].device
        if item.lib is not None:
          # Create runner from cached lib
          prg = get_program(item.ast, Device[device].renderer)
          runner = CompiledRunner(prg, item.lib)
        else:
          # Compile and create runner
          runner = get_runner(device, item.ast)
      elif item.ast.op is Ops.BUFFER_VIEW:
        runner = ViewOp(bufs[0])
      elif item.ast.op is Ops.COPY:
        if hasattr(Device[bufs[0].device].allocator, '_transfer') and all_same([x.device.split(":")[0] for x in bufs]):
          runner = BufferXfer(bufs[0].nbytes, bufs[0].device, bufs[1].device)
        else:
          runner = BufferCopy(bufs[0].nbytes, bufs[0].device, bufs[1].device)
      elif item.ast.op is Ops.ENCDEC:
        runner = EncDec(item.ast, bufs[0].nbytes, bufs[1].device)
      else:
        raise RuntimeError(f"unknown op {item.ast.op}")

      self._bound_items.append((runner, bufs, item.metadata, item.fixedvars))

  def update(self, buffers: dict[UOp, Buffer]|None = None, var_vals: dict[str, int]|None = None) -> ExecutionUnit:
    """Update buffer mapping and/or var_vals for next run. Returns self for chaining."""
    if buffers is not None:
      self.buffer_map.update(buffers)
      # Need to rebind if we update buffers
      self._bound_items = None
    if var_vals is not None:
      self.var_vals.update(var_vals)
    return self

  def __call__(self, var_vals: dict[str, int]|None = None, wait=False, do_update_stats=True, jit=False) -> float|None:
    """Execute all items."""
    from tinygrad.engine.realize import CompiledRunner

    if var_vals is not None:
      self.var_vals.update(var_vals)

    # Lazy bind on first call
    if self._bound_items is None:
      self._bind()

    assert self._bound_items is not None

    # TODO: create graphs on first run
    # if self._first_run:
    #   self._create_graphs()
    #   self._first_run = False

    # Execute all items
    total_et = 0.0
    for runner, bufs, metadata, fixedvars in self._bound_items:
      merged_var_vals = self.var_vals | fixedvars

      # Ensure buffers are allocated (skip if jit - already allocated)
      if not jit:
        for b in bufs:
          b.ensure_allocated()

      # Reorder bufs to match program globals if needed
      if isinstance(runner, CompiledRunner):
        ordered_bufs = [bufs[i] for i in runner.p.globals]
      else:
        ordered_bufs = bufs

      # PROFILE events
      if PROFILE:
        payload = {"metadata":metadata, "var_vals":merged_var_vals, "bufs":[b.trace_num for b in ordered_bufs], "name":runner.display_name}
        payload["outputs"], payload["inputs"] = (runner.p.outs, runner.p.ins) if isinstance(runner, CompiledRunner) else ([0], [1])
        cpu_events.append(ProfilePointEvent(runner.device, "exec", len(cpu_events), payload))

      et = runner(ordered_bufs, merged_var_vals, wait=wait or DEBUG >= 2)
      if et is not None:
        total_et += et

      # Update stats
      if do_update_stats:
        GlobalCounters.kernel_count += 1
        op_est = sym_infer(runner.estimates.ops, merged_var_vals)
        mem_est = sym_infer(runner.estimates.mem, merged_var_vals)
        GlobalCounters.global_ops += op_est
        GlobalCounters.global_mem += mem_est
        if et is not None:
          GlobalCounters.time_sum_s += et
        if DEBUG >= 2:
          lds_est = sym_infer(runner.estimates.lds, merged_var_vals)
          mem_est = min(mem_est, lds_est)   # there can't be more memory accessed than loads/stores
          header_color = 'magenta' if jit else ('green' if runner.first_run else None)
          ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
          flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
          flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
          mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
            colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
          print(f"{colored(f'*** {runner.device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
            f" {runner.display_name+' '*(46-ansilen(runner.display_name))} arg {len(ordered_bufs):2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
            ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
            f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in metadata] if metadata else ''}")
        runner.first_run = False

    return total_et if wait else None

  def __add__(self, other: ExecutionUnit) -> ExecutionUnit:
    """Combine two ExecutionUnits, rebuild graph lazily."""
    combined = ExecutionUnit(self.items + other.items)
    combined.buffer_map = {**self.buffer_map, **other.buffer_map}
    combined.var_vals = {**self.var_vals, **other.var_vals}
    return combined

  def free_intermediates(self):
    """Deallocate internal buffers."""
    for buf in self.buffer_map.values():
      if buf.is_allocated():
        buf.deallocate()
    # Reset bound state
    self._bound_items = None
    self._graphs = None
    self._first_run = True
