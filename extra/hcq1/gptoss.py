"""Prioritize completion of the next GPT-OSS embedding weight, preserving every buffer hazard."""
import functools, hashlib
from tinygrad.device import DepsTracker
from tinygrad.uop.ops import Ops
from extra.hcq1 import schedule

# Inspected native source ABIs: embedding_fwd.cpp reads weight in slot 2, lmhead_ce_bf16.cpp in slot 5.
# Unrecognized sources receive no priority. Kernel names/metadata do not identify the weight's role.
EMBED = 'e2ca63b1c08ee6a93167e2b58b74472d2074bd20c4266183834ab1477622a29f'
LMHEAD = 'bd3a22c965e4dde5e55d43159fdd007bcdd20b70a1ee671ce228321f94d0fab2'
ADAM = 'fused_adam_bf16_vocab_46172160_rawclip1'

def key(b): return b['base'], b['offset'], b['device']

def overlaps(a, b):
  return a['base'] == b['base'] and a['device'] == b['device'] and \
    a['offset'] < b['offset']+b['nbytes'] and b['offset'] < a['offset']+a['nbytes']

def input_producers(rows, r):
  # Follow writes feeding inputs, not old readers or write-after-write scratch hazards.
  for j in r['deps']:
    producer = rows[j]
    if producer['name'] == ADAM: outputs = producer['buffers'][:3]
    elif producer['transport']: outputs = producer['buffers'][:1]
    else: continue
    if any(overlaps(a, b) for a in r['buffers'][1:] for b in outputs): yield j

@functools.cache
def source(ast):
  return hashlib.sha256(ast.src[2].arg.encode()).hexdigest() if ast.op is Ops.PROGRAM else None

def records(graph):
  tracker, result = DepsTracker(), []
  for i, (_, ast, bufs, _) in enumerate(graph.calls):
    result.append(dict(index=i, op=ast.op.name, name=ast.arg.name if ast.op is Ops.PROGRAM else None,
      source=source(ast), deps=sorted(set(tracker.access_resources(bufs, schedule.write_slots(ast, bufs), i))),
      transport=schedule.is_vocab_transport(ast, bufs),
      gather=ast.op is Ops.PROGRAM and len(bufs) == 9 and bufs[0].nbytes == 738754560 and
        all(b.nbytes == 92344320 for b in bufs[1:]) and schedule.is_bf16_vocab_gather(ast),
      buffers=[dict(base=id(b.base), offset=b.offset, nbytes=b.nbytes, device=b.device) for b in bufs]))
  return result

def embedding_gather_producers(rows, copies):
  found = set()
  for reader in rows:
    if reader['source'] != EMBED: continue
    stack, visited = [(reader, reader['buffers'][2])], set()
    while stack:
      consumer, weight = stack.pop()
      for j in consumer['deps']:
        p = rows[j]
        if j in visited or not (p['gather'] or j in copies): continue
        if key(p['buffers'][0]) != key(weight) or p['buffers'][0]['nbytes'] != weight['nbytes']: continue
        visited.add(j)
        if p['gather']: found.add(j)
        else: stack.append((p, p['buffers'][1]))
  return found

def embedding_completion_mask(rows, copies, roles):
  for r in rows:
    if r['source'] not in (EMBED, LMHEAD): continue
    slot, role = (2, 'embedding') if r['source'] == EMBED else (5, 'lm_head')
    b = r['buffers'][slot]
    if b['nbytes'] != 738754560: continue
    assert roles.get(key(b), role) == role
    roles[key(b)] = role
  early, known = set(), set()
  for r in rows:
    if not r['gather'] or (role := roles.get(key(r['buffers'][0]))) is None: continue
    known.add(r['index'])
    if role != 'embedding': continue
    selected, stack = set(), list(input_producers(rows, r))
    while stack:
      i = stack.pop()
      if i in selected: continue
      if rows[i]['transport'] or rows[i]['name'] == ADAM:
        selected.add(i)
        if rows[i]['transport']: stack.extend(input_producers(rows, rows[i]))
    early.update(selected)
  for g in sorted(embedding_gather_producers(rows, copies)-known):
    selected, stack = set(), list(input_producers(rows, rows[g]))
    while stack:
      j = stack.pop()
      if j in selected: continue
      selected.add(j)
      if rows[j]['transport']: stack.extend(input_producers(rows, rows[j]))
    early.update(selected)
  return [i in early for i in range(len(rows))]

# Preserve physical identities across capture batches, including small graphs which only register a reader.
_roles, _held_bases = {}, set()

def reorder_embedding(graph):
  rows = records(graph)
  copies = {i for i, (_, ast, bufs, _) in enumerate(graph.calls) if len(bufs) == 2 and
            bufs[0].nbytes == bufs[1].nbytes == 738754560 and ast.op is Ops.PROGRAM and schedule.is_storage_copy(ast)}
  eager = embedding_completion_mask(rows, copies, _roles)
  for _, _, bufs, _ in graph.calls:
    for b in bufs:
      if (id(b.base), b.offset, b.device) in _roles: _held_bases.add(b.base)
  if len(rows) < 1000 or not any(eager): return
  order = schedule.eager_ready_order([r['deps'] for r in rows], list(range(len(rows))), eager)
  positions = {old:new for new, old in enumerate(order)}
  for field in ('calls', 'runtimes', 'uop_replace'):
    old = getattr(graph, field)
    setattr(graph, field, [old[i] for i in order])
  for field in ('var_vals_replace', 'launch_dims_replace', 'launch_dims_base'):
    setattr(graph, field, {positions[k]:v for k, v in getattr(graph, field).items()})
