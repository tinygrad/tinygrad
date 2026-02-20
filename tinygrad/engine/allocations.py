from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, GroupOp, graph_rewrite, _remove_all_tags
from tinygrad.dtype import ImageDType
from tinygrad.helpers import prod, DEBUG, argsort

# these are the only uops that can get replaced in the tensor graph
from tinygrad.schedule.rangeify import pm_gate_kernel_sink

def tag_uop(ctx:tuple[list[UOp], set[UOp], dict[UOp, UOp], set[UOp]], x:UOp):
  if x.tag is not None or x in ctx[1]: return None
  if x.tag is None and x.op is Ops.CALL:
    # don't tag anything in a CALL
    for u in x.src[0].toposort(): ctx[1].add(u)
  ctx[0].append(x)
  return x.replace(tag=(len(ctx[0])-1,))

def disk_copy_is_buffer(ctx, u):
  # copies to disk are replaced with the disk buffer
  to_disk = isinstance(u._device, str) and u._device.startswith("DISK")
  if to_disk: ctx[2][u] = UOp.new_buffer(u.device, u.shard_size, u.dtype).reshape(u.max_shard_shape)
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0]._device, str) and any(u.src[0]._device.startswith(x) for x in ["NPY", "DISK"])
  if from_creation: return tag_uop(ctx, u)

def apply_after(ctx, u):
  ctx[2][u] = u.src[0]

# CONTIGUOUS and ASSIGN + parents are the only nodes that get updated
add_tags = pm_gate_kernel_sink+PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat({Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx[3] else None),
])

def replace_contig_with_assign(u:UOp):
  # no real contig for DISK tensors, they are left alone
  if isinstance(u._device, str) and u._device.startswith("DISK"): return u.rtag(None)
  dtype = u.dtype
  if isinstance(dtype, ImageDType):
    if prod(dtype.shape) != prod(u.max_shard_shape) or ([x for x in u.max_shard_shape if x != 1] or [1])[-1] % 4 != 0:
      if DEBUG >= 1: print(f"demoting Image {dtype} with shape {u.max_shard_shape}")
      dtype = dtype.base
  buffer = UOp.new_buffer(u.device, u.shard_size, dtype).reshape(u.max_shard_shape)
  if isinstance(u.device, tuple) and u.axis is not None: buffer = buffer.multi(u.axis)
  return buffer.assign(u.src[0]).rtag(u.tag)

def replace_assign_with_contig(u:UOp):
  assigned_to = u
  while assigned_to.op in {Ops.ASSIGN, Ops.BITCAST}: assigned_to = assigned_to.src[0].base
  if assigned_to.op is not Ops.BUFFER:
    return u.src[1].contiguous(tag=u.tag)

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  x = src
  while x is not src.base:
    if x.op is Ops.PERMUTE: contig = contig.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: contig = contig.reshape(x.src[0].shape)
    else: return None
    x = x.src[0]
  ctx[src.base] = contig

pm_early_transform_tensor_graph = PatternMatcher([
  # CONTIGUOUS replacement hack for openpilot
  (UPat(Ops.CONTIGUOUS, src=(UPat(GroupOp.Movement, name="src"),), name="contig"), found_contiguous),
  # replace ALU sources with contiguous versions found above
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
  # add CONTIGUOUS to tagged UOps
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag is not None else None),
  # remove extra CONTIGUOUS on ASSIGN
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.ASSIGN, name="a"),), name="c"), lambda a,c: a.replace(tag=a.tag+c.tag)),
  # replace ASSIGN with CONTIGUOUS
  (UPat(Ops.ASSIGN, name="u"), replace_assign_with_contig),
  # replace CONTIGUOUS with ASSIGNs
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_assign),
  # just removing it works...
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
])

def allocate_global_buffers(big_sink:UOp) -> tuple[UOp, dict[UOp, UOp]]:
  # uop list is a list in the original_sink graph and we can map to the tags later
  # here we build buffer map
  uop_list: list[UOp] = []
  buffer_map: dict[UOp, UOp] = {}

  dont_realize = {Ops.CONST, Ops.BUFFER, Ops.BIND, Ops.DEFINE_VAR, Ops.AFTER}
  bases = set([x.multibase for x in big_sink.src if x.base.op not in dont_realize])

  # this rewrite is "read-only", it adds simple things to buffer_map and may sink things on big_sink, bottom_up
  # this is the only one where we have to be careful to not break the tensor graph
  big_sink = graph_rewrite(big_sink, add_tags, ctx=(uop_list, set(), buffer_map, bases), bottom_up=True, name="number the uops")

  # here we can break the tensor graph. this is the only place you need to maintain numbered tags
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, ctx={}, name="early transform tensor graph")

  # here we construct the final buffer_map. this is everything that will go into the tensor map
  for s in big_sink.toposort():
    if s.tag is not None:
      assert s.op is Ops.ASSIGN
      for t in s.tag:
        original_uop = uop_list[t]
        replace_uop = s
        while replace_uop.op is Ops.ASSIGN: replace_uop = replace_uop.src[0]
        buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  big_sink = graph_rewrite(big_sink, _remove_all_tags, name="remove tags")
  return big_sink, buffer_map
