from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
import math, functools
from typing import List, Dict, Callable, Tuple, Type, Union
from tinygrad.helpers import partition, all_same

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

class Node:
  b: int
  min: int
  max: int
  def render(self, ops=None, ctx=None) -> str:
    if ops is None: ops = render_python
    assert isinstance(self, (Variable, NumNode)) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  def __repr__(self): return "<"+self.key+">"
  def __eq__(self, other:object) -> bool:
    if not isinstance(other, Node): return NotImplemented
    return self.key == other.key
  def __neg__(self): return self*-1
  def __add__(self, b:Union[Node, int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  def __sub__(self, b:Union[Node, int]): return self+-b
  def __ge__(self, b:int): return create_node(GeNode(self, b))
  def __lt__(self, b:int): return create_node(LtNode(self, b))
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    return create_node(MulNode(self, b))

  # *** complex ops ***

  def __floordiv__(self, b:int, factoring_allowed=True):
    assert b != 0
    if b < 0: return (self//-b)*-1
    if b == 1: return self

    # the numerator of div is not allowed to be negative
    if self.min < 0:
      offset = self.min//b
      # factor out an "offset" to make the numerator positive. don't allowing factoring again
      return (self + -offset*b).__floordiv__(b, factoring_allowed=False) + offset
    return create_node(DivNode(self, b))

  def __mod__(self, b:int):
    assert b > 0
    if b == 1: return NumNode(0)
    if self.min >= 0 and self.max < b: return self
    if self.min < 0: return (self - ((self.min//b)*b)) % b
    return create_node(ModNode(self, b))

  @staticmethod
  def num(num:int) -> Node: return NumNode(num)

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    new_nodes: List[Node] = []
    sum_nodes: List[SumNode] = []
    num_nodes: List[NumNode] = []
    mul_nodes: List[MulNode] = []
    for node in nodes:
      if isinstance(node, NumNode):
        num_nodes.append(node)
      elif isinstance(node, MulNode):
        mul_nodes.append(node)
      elif isinstance(node, SumNode): # expand any sums inside one sum
        sum_nodes.append(node)
      else:
        new_nodes.append(node)

    # expand any sums inside one sum
    if sum_nodes:
      new_nodes.extend(num_nodes)
      new_nodes.extend(mul_nodes)
      for x in sum_nodes: new_nodes += x.nodes
      return Variable.sum(new_nodes)

    # combine any numbers inside a sum
    if num_nodes:
      new_nodes.append(NumNode(sum([x.b for x in num_nodes])))

    # combine any MulNodes that factorize (big hack sticking the MulNode(x, 1) on things)
    mul_nodes += [MulNode(x, 1) for x in new_nodes]
    mul_groups: Dict[str, Tuple[Node, List[MulNode]]] = defaultdict(lambda: (Node(), []))
    for node in mul_nodes: #NOTE can we somehow avoid rendering here?
      key = node.a.render()
      mul_groups[key] = (node.a, mul_groups[key][1] + [node])
    mul_nodes = [k * sum(x.b for x in g) for k, g in mul_groups.values()]
    new_nodes = [x if not isinstance(x, MulNode) or x.b != 1 else x.a for x in mul_nodes]

    # filter 0s
    new_nodes = [x for x in new_nodes if x.min != 0 or x.max != 0]
    return create_rednode(SumNode, new_nodes) if len(new_nodes) > 1 else (new_nodes[0] if len(new_nodes) == 1 else NumNode(0))

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_rednode(AndNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __new__(cls, expr:str, nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)

  def __init__(self, expr:str, nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num

def create_node(ret:Node):
  assert ret.min <= ret.max, f"min greater than max! {ret.min} {ret.max} when creating {type(ret)} {ret}"
  if ret.min == ret.max: return NumNode(ret.min)
  return ret

class OpNode(Node):
  def __init__(self, a:Node, b:int): 
    self.a, self.b = a, b
    self.min, self.max = self.get_bounds()
  @abstractmethod 
  def get_bounds(self) -> Tuple[int, int]: pass

class GeNode(OpNode):
  def __mul__(self, b: int): return (self.a*b) >= (self.b*b)
  def __floordiv__(self, b: int, _=False): return (self.a//b) >= (self.b//b)
  def get_bounds(self) -> Tuple[int, int]: return int(self.a.min >= self.b), int(self.a.max >= self.b)
class LtNode(OpNode):
  def __mul__(self, b: int): return (self.a*b) < (self.b*b)
  def __floordiv__(self, b: int, _=False): return (self.a//b) < (self.b//b)
  def get_bounds(self) -> Tuple[int, int]: return int(self.a.max < self.b), int(self.a.min < self.b)
class MulNode(OpNode):
  def __mul__(self, b: int): return self.a*(self.b*b) # two muls in one mul
  def __floordiv__(self, b: int, factoring_allowed=False): # NOTE: mod negative isn't handled right
    if self.b % b == 0: return self.a*(self.b//b)
    if b % self.b == 0 and self.b > 0: return self.a//(b//self.b)
    return Node.__floordiv__(self, b, factoring_allowed)
  def __mod__(self, b: int):
    a = (self.a * (self.b%b))
    return Node.__mod__(a, b)
  def get_bounds(self) -> Tuple[int, int]: 
    return (self.a.min*self.b, self.a.max*self.b) if self.b >= 0 else (self.a.max*self.b, self.a.min*self.b)
class DivNode(OpNode):
  def __floordiv__(self, b: int, _=False): return self.a//(self.b*b) # two divs is one div
  def get_bounds(self) -> Tuple[int, int]: 
    assert self.a.min >= 0
    return self.a.min//self.b, self.a.max//self.b
class ModNode(OpNode):
  def __floordiv__(self, b: int, factoring_allowed=True):
    if (self.b % b == 0): return (self.a//b) % (self.b//b) # put the div inside mod
    return Node.__floordiv__(self, b, factoring_allowed)
  def get_bounds(self) -> Tuple[int, int]: 
    assert self.a.min >= 0
    return (0, self.b-1) if self.a.max - self.a.min >= self.b or (self.a.min != self.a.max and self.a.min%self.b >= self.a.max%self.b) else (self.a.min%self.b, self.a.max%self.b)

class RedNode(Node):
  def __init__(self, nodes:List[Node]): self.nodes = nodes

class SumNode(RedNode):
  def __mul__(self, b: int): return Variable.sum([x*b for x in self.nodes]) # distribute mul into sum
  def __floordiv__(self, b: int, factoring_allowed=True):
    if not factoring_allowed: return Node.__floordiv__(self, b, factoring_allowed)
    factors, tmp_nofactor = partition(self.nodes, lambda x: (isinstance(x, (MulNode, NumNode))) and x.b%b == 0)
    nofactor = []
    # ugh, i doubt this is universally right
    for x in tmp_nofactor:
      if isinstance(x, NumNode):
        if (x.b%b) != x.b:
          factors.append(Variable.num(x.b - (x.b%b)))  # python does floor division
        nofactor.append(Variable.num(x.b%b))
      else:
        nofactor.append(x)
    gcd = [math.gcd(x.b, b) if isinstance(x, (MulNode, NumNode)) else None for x in nofactor]
    if len(factors) > 0:
      # these don't have to be the same, just having a common factor
      if len(gcd) > 0 and all_same(gcd) and gcd[0] is not None and gcd[0] > 1:
        nofactor_term = Variable.sum([(x.a * (x.b//gcd[0])) if isinstance(x, MulNode) else Variable.num(x.b//gcd[0]) for x in nofactor])//(b//gcd[0])
      else:
        nofactor_term = Variable.sum(nofactor)//b
      return Variable.sum([(x.a * (x.b//b)) if isinstance(x, MulNode) else Variable.num(x.b//b) for x in factors] + [nofactor_term])
    else:
      muls = [x.b for x in nofactor if isinstance(x, MulNode)]
      for m in muls:
        if m > 1 and b%m == 0:
          return (self//m)//(b//m)
      return Node.__floordiv__(self, b, factoring_allowed)
  def __mod__(self, b: int): 
    new_nodes = []
    for x in self.nodes:
      if isinstance(x, NumNode): new_nodes.append(Variable.num(x.b%b))
      elif isinstance(x, MulNode): new_nodes.append(x.a * (x.b%b))
      else: new_nodes.append(x)
    return Node.__mod__(Variable.sum(new_nodes), b)

class AndNode(RedNode): 
  def __mul__(self, b: int): Variable.ands([x*b for x in self.nodes])
  def __floordiv__(self, b: int, _=True): return Variable.ands([x//b for x in self.nodes])

def create_rednode(typ:Type[RedNode], nodes:List[Node]):
  ret = typ(nodes)
  if typ == SumNode: ret.min, ret.max = (sum([x.min for x in nodes]), sum([x.max for x in nodes]))
  elif typ == AndNode: ret.min, ret.max = (min([x.min for x in nodes]), max([x.max for x in nodes]))
  return create_node(ret)

render_python: Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{self.b})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  GeNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}>={self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{self.b})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"
}