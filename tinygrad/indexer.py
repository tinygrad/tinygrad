from typing import List

class Node:
  pass

class NumNode(Node):
  def __init__(self, num:int):
    self.num = num
  def __str__(self):
    return str(self.num)

class VariableNode(Node):
  def __init__(self, expr:str, min:int, max:int):
    self.expr, self.min, self.max = expr, min, max
  def __str__(self):
    return self.expr

class SumNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
  def __str__(self):
    return f"({'+'.join([str(x) for x in self.nodes])})"

class MulNode(Node):
  def __init__(self, a:VariableNode, b:int):
    self.a, self.b = a, b
  def __str__(self):
    return f"({self.a}*{self.b})"

class DivNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
  def __str__(self):
    return f"({self.a}/{self.b})"

class ModNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
  def __str__(self):
    return f"({self.a}%{self.b})"

