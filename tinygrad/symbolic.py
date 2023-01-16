from typing import List, Optional

class Node:
  pass

class Variable(Node):
  def __init__(self, expr:str, min:int, max:int):
    self.expr, self.min, self.max = expr, min, max
  @staticmethod
  def num(num:int):
    return Variable(str(num), num, num)
  def __str__(self):
    return self.expr
  def __add__(self, num:int): return AddNode(self, num)
  def __mul__(self, num:int): return MulNode(self, num)
  def __floordiv__(self, num:int): return DivNode(self, num)
  def __mod__(self, num:int): return ModNode(self, num)
  def __ge__(self, num:int): return GeNode(self, num)
  def __lt__(self, num:int): return LtNode(self, num)

class AddNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min+b, a.max+b
  def __str__(self):
    return f"({self.a}+{self.b})" if self.b != 0 else str(self.a)

class MulNode(Variable):
  def __new__(cls, a:Variable, b:int):
    if b == 0: return Variable.num(0)
    elif b == 1: return a
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min*b, a.max*b
  def __str__(self):
    return f"({self.a}*{self.b})"

class DivNode(Variable):
  def __new__(cls, a:Variable, b:int):
    assert b != 0
    if b == 1: return a
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min//b, a.max//b
  def __str__(self):
    return f"({self.a}//{self.b})"

class ModNode(Variable):
  # TODO: why is this broken?
  def __new__(cls, a:Variable, b:int):
    if b == 1: return Variable.num(0)
    if a.min >= 0 and a.max < b: return a
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = min(a.min, 0), max(a.max, b)
  def __str__(self):
    return f"({self.a}%{self.b})"

class GeNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  def __str__(self):
    return f"({self.a} >= {self.b})"

class LtNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  def __str__(self):
    return f"({self.a} < {self.b})"

# reduce nodes

class SumNode(Variable):
  def __init__(self, nodes:List[Variable]):
    self.nodes = nodes
    self.min, self.max = sum([x.min for x in nodes]), sum([x.max for x in nodes])
  def __str__(self):
    return f"({'+'.join(['0']+[str(x) for x in self.nodes if str(x) != '0'])})"

class AndNode(Variable):
  def __init__(self, nodes:List[Variable]):
    self.nodes = nodes
    self.min, self.max = min([x.min for x in nodes]), max([x.max for x in nodes])
  def __str__(self):
    return f"({'&&'.join([str(x) for x in self.nodes])})"

