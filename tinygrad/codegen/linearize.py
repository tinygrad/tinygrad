from __future__ import annotations
import collections, heapq, functools
from dataclasses import dataclass
from tinygrad.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite, GroupOp
from tinygrad.spec import type_verify
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import dedup, flatten, partition

DONT_PLACE_IN_BLOCK = {Ops.NAME, Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST, *GroupOp.Block}

# root_uop -> sink
def linearize_uop(root_uop:UOp, skip_check:bool=not __debug__) -> list[UOp]:
  scope_blocks = {}

  class LinearizeScope:
    def __init__(self):
      self.define_acc = []
      self.start = []
      self.end = []
      self.context = []
      self.body = []
      self.subgraph_nodes = set()
    
    def fuse(self, other):
      scope = LinearizeScope()
      scope.define_acc = dedup(self.define_acc + other.define_acc)
      scope.start = dedup(self.start + other.start)
      scope.end = dedup(self.end + other.end)
      scope.body = dedup(self.body + other.body)
      scope.subgraph_nodes = set.union(self.subgraph_nodes, other.subgraph_nodes)

      for elem in scope.subgraph_nodes:
        for src in elem.src:
          if src not in scope.subgraph_nodes:
            scope.context.append(src)

      body_set = set(scope.body)
      for i in scope.start + scope.end:
        assert i not in body_set, "Broken skopes"

      return scope

    @functools.cached_property
    def linear(self):
      linear = []

      for elem in self.define_acc:
        linear.append(elem)

      for elem in self.end:
        linear.append(elem)
      
      curr_context = set(self.context + self.define_acc + self.end)
      simple_set = set([elem for elem in self.body if elem.op not in {Ops.ASSIGN, Ops.STORE}])
      scope_set = set([elem for elem in self.body if elem.op in {Ops.ASSIGN, Ops.STORE}])
      fusion_set = set()
      poped = True
      while True:
        poped = False
        pop_list = []
        for elem in simple_set:
          ready = True
          for cont in elem.src:
            if cont not in curr_context:
              ready = False
              break
          
          if ready:
            pop_list.append(elem)
            curr_context.add(elem)
            if self.start[0].op is Ops.SINK:
              for s in elem.src:
                assert elem.op is Ops.DEFINE_ACC or s in linear
            linear.append(elem)
            poped = True

        for elem in pop_list:
          simple_set.remove(elem)
        
        if poped:
          continue

        for elem in scope_set:
          elem_scope = scope_blocks[elem]
          ready = True
          for cont in elem_scope.context:
            if cont not in curr_context:
              ready = False
              break

          if ready:
            fusion_set.add(elem)
        
        for elem in fusion_set:
          scope_set.discard(elem)
        
        if not fusion_set:
          assert not simple_set and not scope_set, "Linearization Failed"
          break

        fusion_elem = fusion_set.pop()
        fusion_elem = scope_blocks[fusion_elem]
        for cont_elem in fusion_elem.subgraph_nodes:
          curr_context.add(cont_elem)
        pop_list = []
        for elem in fusion_set:
          scope_elem = scope_blocks[elem]
          fusable = True
          for i in scope_elem.end:
            if i not in fusion_elem.end:
              fusable = False

          for i in fusion_elem.end:
            if i not in scope_elem.end:
              fusable = False
          
          if not fusable:
            continue

          fusion_elem = fusion_elem.fuse(scope_elem)
          for cont_elem in scope_elem.subgraph_nodes:
            curr_context.add(cont_elem)

          pop_list.append(elem)
        
        for elem in pop_list:
          fusion_set.remove(elem)
        
        linear = linear + fusion_elem.linear

        for e in fusion_elem.subgraph_nodes:
          assert e in fusion_elem.linear

      for elem in self.start:
        linear.append(elem)

      if self.start[0].op in {Ops.ASSIGN, Ops.STORE}:
        for elem in self.end[::-1]:
          linear.append(UOp(Ops.ENDRANGE if elem.op is Ops.RANGE else Ops.ENDIF, src=(elem,)))

      if self.start[0].op is Ops.SINK:
        for i, o in enumerate(linear):
          for s in o.src:
            assert o.op is Ops.DEFINE_ACC or s in linear[:i]

      linear = sorted([elem for elem in root_uop.toposort if elem.op is Ops.DEFINE_GLOBAL], key=lambda x: x.arg) + [elem for elem in linear if elem.op is not Ops.DEFINE_GLOBAL]
      if self.start[0].op is Ops.SINK:
        assert root_uop.arg is not None
        linear = [UOp(Ops.NAME, arg=root_uop.arg.name)] + linear

      return linear

  def _make_scope(scope_start:UOp):
    scope = LinearizeScope()
    scope.start.append(scope_start)
    stack = [(scope_start, None, False)]
    scope.subgraph_nodes.add(scope_start)
    visited = set()
    while stack:
      curr_uop, pare_uop, bubbling = stack.pop()
      if curr_uop in visited:
        if curr_uop in scope.subgraph_nodes and pare_uop is not None:
          scope.subgraph_nodes.add(pare_uop)
        continue
      elif bubbling:
        visited.add(curr_uop)

      if bubbling:
        if curr_uop.op in {Ops.RANGE, Ops.IF} and scope_start.op in {Ops.ASSIGN, Ops.STORE} or \
           curr_uop.src == () and scope_start.op == Ops.SINK:
          scope.end.append(curr_uop)
          scope.subgraph_nodes.add(curr_uop)

        if curr_uop in scope.subgraph_nodes and pare_uop is not None:
          scope.subgraph_nodes.add(pare_uop)
          if curr_uop.key != scope_start.key and curr_uop.op not in {Ops.RANGE, Ops.IF} and \
            not (curr_uop.src == () and scope_start.op == Ops.SINK):
            if curr_uop.op is Ops.DEFINE_ACC:
              scope.define_acc.append(curr_uop)
            else:
              scope.body.append(curr_uop)
      else:
        stack.append((curr_uop, pare_uop, True))
        if curr_uop.op not in {Ops.ASSIGN, Ops.STORE, Ops.RANGE, Ops.IF} or pare_uop is None:
          for uop_src in curr_uop.src:
            stack.append((uop_src, curr_uop, False))
        elif curr_uop.op in {Ops.ASSIGN, Ops.STORE}:
          for uop_src in scope_blocks[curr_uop].context:
            stack.append((uop_src, curr_uop, False))
    
    for elem in scope.body:
      assert elem.op not in {Ops.RANGE, Ops.IF}, "Colliding scopes"
      if elem.op in {Ops.ASSIGN, Ops.STORE}:
        scope.subgraph_nodes = set.union(scope.subgraph_nodes, scope_blocks[elem].subgraph_nodes)

    for elem in scope.subgraph_nodes:
      for src in elem.src:
        if src not in scope.subgraph_nodes:
          scope.context.append(src)
    
    scope.context = dedup(scope.context)

    return scope

  scopes_list = [elem for elem in root_uop.toposort if elem.op in {Ops.ASSIGN, Ops.STORE}]
  for scope_start in scopes_list:
    scope = _make_scope(scope_start)
    scope_blocks[scope_start] = scope
    print(" ----- ")
    for i in scope.start:
      print(f"scope start {i.op}")
    print(" -   - ")
    for i in scope.body:
      print(f"scope elem {i.op}")
    print(" -   - ")
    for i in scope.end:
      print(f"scope end {i.op}")
    print(" -   - ")
    for i in scope.context:
      print(f"scope context {i.op}")
    print(" ----- ")

  scope = _make_scope(root_uop)
  print(" --- global --- ")
  for i in scope.start:
    print(f"scope start {i.op}")
  print(" -   - ")
  for i in scope.body:
    print(f"scope elem {i.op}")
  print(" -   - ")
  for i in scope.end:
    print(f"scope end {i.op}")
  print(" -   - ")
  for i in scope.context:
    print(f"scope context {i.op}")
  print(" --- global --- ")

  print(" --- final --- ")
  for i in scope.linear:
    print(f"scope linear --> {i.op}")
  print(" --- final --- ")
  return scope.linear[:-1]
