from tinygrad.uop.ops import UOp, UPat, PatternMatcher, Ops, GroupOp, graph_rewrite, identity_element
from tinygrad.dtype import ImageDType
from tinygrad.helpers import prod, DEBUG, argsort, VIZ

def tag_uop(ctx:tuple[list[UOp], dict[UOp, UOp], set[UOp]], x:UOp):
  if x.tag is not None: return None
  ctx[0].append(x)
  return x.replace(tag=(len(ctx[0])-1,))

def disk_copy_is_buffer(ctx, u):
  # copies to disk are replaced with the disk buffer
  to_disk = isinstance(u._device, str) and u._device.startswith("DISK")
  if to_disk: ctx[1][u] = UOp.new_buffer(u.device, u.shard_size, u.dtype).reshape(u.max_shard_shape)
  # all copies from disk/numpy are realized into a real buffer
  from_creation = isinstance(u.src[0]._device, str) and any(u.src[0]._device.startswith(x) for x in ["NPY", "DISK", "PYTHON"])
  if from_creation: return tag_uop(ctx, u)

def apply_after(ctx, u):
  ctx[1][u] = u.src[0]

# CONTIGUOUS and ASSIGN + parents are the only nodes that get updated
add_tags = PatternMatcher([
  (UPat(Ops.COPY, name="u"), disk_copy_is_buffer),
  # no tag on copies that are assigned
  (UPat(Ops.ASSIGN, src=(UPat(), UPat(Ops.COPY, name="c")), name="a"),
   lambda a,c: a.replace(src=(a.src[0], c.rtag(())), tag=a.tag+c.tag) if a.tag and c.tag else None),
  (UPat(Ops.AFTER, name="u"), apply_after),
  (UPat({Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), tag_uop),
  (UPat(GroupOp.All, name="x"), lambda ctx,x: tag_uop(ctx,x) if x in ctx[2] else None),
])

def replace_contig_with_assign(u:UOp):
  # if size is 0, remove the contig
  if u.size == 0: return u.src[0]
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
  (UPat(GroupOp.All-{Ops.CONTIGUOUS, Ops.ASSIGN}, name="x"), lambda x: x.rtag(None).contiguous(tag=x.tag) if x.tag else x.replace(tag=None)),
  # remove extra CONTIGUOUS on ASSIGN
  (UPat(Ops.CONTIGUOUS, src=(UPat(Ops.ASSIGN, name="a"),), name="c"), lambda a,c: a.replace(tag=a.tag+c.tag)),
  # replace ASSIGN with CONTIGUOUS
  (UPat(Ops.ASSIGN, name="u"), replace_assign_with_contig),
  # replace CONTIGUOUS with ASSIGNs
  (UPat(Ops.CONTIGUOUS, name="u"), replace_contig_with_assign),
  # remove DETACH/CONTIGUOUS_BACKWARD
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and x.size == 0 else None),
  # early fixup const copy (TODO: is this wrong if there's a pad?)
  (UPat(Ops.COPY, src=(UPat.var("s"), UPat()), name="c"), lambda c,s: c.const_like(ss.arg) if (ss:=s.base).op is Ops.CONST else None),
])

def untag_and_append(ctx:tuple[list[UOp], dict[UOp, UOp], list[UOp]], x:UOp):
  if x.tag is None: return None
  uop_list, buffer_map, assigns = ctx
  ret = x.replace(tag=None)
  for t in x.tag:
    original_uop: UOp = uop_list[t]
    replace_uop = ret
    while replace_uop.op is Ops.ASSIGN: replace_uop = replace_uop.src[0]
    buffer_map[original_uop] = replace_uop.shrink_to(original_uop.shape)
  assigns.append(ret)
  return ret

def append_after(ctx:tuple[list[UOp], dict[UOp, UOp], list[UOp]], x:UOp):
  ctx[2].append(x)

pm_finalize_call = PatternMatcher([
  (UPat(Ops.ASSIGN, name="x"), untag_and_append),
  (UPat(Ops.AFTER, name="x"), append_after),
  (UPat(Ops.COPY, name="x"), lambda ctx,x: append_after(ctx,x) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # replace UNIQUE with LUNIQUE for CONST cache key normalization
  (UPat(Ops.CONST, src=(UPat(Ops.UNIQUE), UPat(Ops.DEVICE, name="d")), name="b"), lambda b,d: b.replace(src=(d,))),
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
  big_sink = graph_rewrite(big_sink, add_tags, ctx=(uop_list, buffer_map, bases), bottom_up=True, name="number the uops")

  # here we can break the tensor graph. this is the only place you need to maintain numbered tags
  big_sink = graph_rewrite(big_sink, pm_early_transform_tensor_graph, ctx={}, name="early transform tensor graph")

  # here we construct the final buffer_map. this is everything that will go into the tensor map
  assigns: list[UOp] = []
  graph_rewrite(big_sink, pm_finalize_call, ctx=(uop_list, buffer_map, assigns), name="finalize call")
  ret = UOp.sink(*assigns)
  if VIZ: graph_rewrite(ret, PatternMatcher([]), name="*** Call")
  return ret, buffer_map
