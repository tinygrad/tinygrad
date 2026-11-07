# **** op specializer ****
# Partially evaluates a function on a constant op: `match op:` cases and tests that only depend on the op (and globals) are resolved
# at compile time, the rest is left as is. This is used for the per-op rules (dtype, shape), so they are written once as a big `match`,
# but a lookup is a dict access and a call to a small function.
import ast, inspect, textwrap, copy
from typing import Any, Callable, cast

class _Static(Exception): pass

def _same(a:ast.AST, b:ast.AST) -> bool:
  # structural equality for the op expression (a Name or an Attribute chain)
  if isinstance(a, ast.Name) and isinstance(b, ast.Name): return a.id == b.id
  if isinstance(a, ast.Attribute) and isinstance(b, ast.Attribute): return a.attr == b.attr and _same(a.value, b.value)
  return False

class _OpSpecializer:
  def __init__(self, fxn:Callable, op_expr:str):
    self.fxn, self.globs = fxn, fxn.__globals__  # type: ignore[attr-defined]
    src = textwrap.dedent(inspect.getsource(fxn))
    self.tree = ast.parse(src)
    ast.increment_lineno(self.tree, fxn.__code__.co_firstlineno-1)  # type: ignore[attr-defined]
    fdef = self.tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    self.fdef: ast.FunctionDef = fdef
    self.fdef.decorator_list = []
    self.op_expr = ast.parse(op_expr, mode="eval").body
    # names that are not globals: the args and anything assigned in the function
    self.local_names = {a.arg for a in self.fdef.args.args} | \
                       {n.id for n in ast.walk(self.fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    self.static_cache: dict[tuple[int, Any], Any] = {}

  def static_eval(self, node:ast.expr, op) -> Any:
    """evaluate an expression that only depends on the op, raises _Static if it can't"""
    if (key:=(id(node), op)) not in self.static_cache: self.static_cache[key] = self._static_eval(node, op)
    # NOTE: raise a new exception, raising the cached one would attach the traceback (and all its frames) to the cache
    if isinstance(ret:=self.static_cache[key], _Static): raise _Static()
    return ret
  def _static_eval(self, node:ast.expr, op) -> Any:
    op_expr = self.op_expr
    # anything that reads a local, other than through the op expression, isn't static
    def reads_local(n:ast.AST) -> bool:
      if _same(n, op_expr): return False
      if isinstance(n, ast.Name) and n.id in self.local_names: return True
      return any(reads_local(c) for c in ast.iter_child_nodes(n))
    if reads_local(node): return _Static()
    class Sub(ast.NodeTransformer):
      def visit(s, n):
        if _same(n, op_expr): return ast.copy_location(ast.Name("__OP__", ast.Load()), n)
        return super().visit(n)
    expr = ast.Expression(Sub().visit(copy.deepcopy(node)))
    try: return eval(compile(ast.fix_missing_locations(expr), "<static>", "eval"), self.globs, {"__OP__": op})  # pylint: disable=eval-used
    except Exception: return _Static()

  def pattern_matches(self, pat:ast.pattern, op) -> bool:
    if isinstance(pat, ast.MatchValue): return bool(op == self.static_eval(pat.value, op))
    if isinstance(pat, ast.MatchOr): return any(self.pattern_matches(p, op) for p in pat.patterns)
    if isinstance(pat, ast.MatchAs) and pat.pattern is None and pat.name is None: return True  # wildcard
    raise _Static()

  def stmts(self, body:list[ast.stmt], op) -> list[ast.stmt]:
    """resolve the static `match` and `if` statements. NOTE: this doesn't modify the input"""
    ret: list[ast.stmt] = []
    for st in body:
      # anything after a return is dead
      if ret and isinstance(ret[-1], (ast.Return, ast.Raise)): break
      if isinstance(st, ast.Match) and _same(st.subject, self.op_expr):
        try:
          if any(c.guard is not None for c in st.cases): raise _Static()
          # the matched case runs, then the code after the match (if the case doesn't return)
          if (case:=next((c for c in st.cases if self.pattern_matches(c.pattern, op)), None)) is not None: ret.extend(self.stmts(case.body, op))
          continue
        except _Static: pass
      if isinstance(st, ast.If):
        try:
          ret.extend(self.stmts(st.body if self.static_eval(st.test, op) else st.orelse, op))
          continue
        except _Static:
          st = ast.copy_location(ast.If(st.test, self.stmts(st.body, op) or [ast.Pass()], self.stmts(st.orelse, op)), st)
      elif isinstance(st, ast.Match):
        cases = [ast.match_case(c.pattern, c.guard, self.stmts(c.body, op) or [ast.Pass()]) for c in st.cases]
        st = ast.copy_location(ast.Match(st.subject, cases), st)
      ret.append(st)
    return ret

  def exprs(self, tree:ast.AST, op) -> ast.AST:
    """fold the op dependent tests inside expressions: `and`/`or` operands and `a if test else b`. NOTE: this modifies the input"""
    spec = self
    def static(n:ast.expr) -> tuple[bool, Any]:
      try: return True, spec.static_eval(n, op)
      except _Static: return False, None
    class Fold(ast.NodeTransformer):
      def visit_IfExp(s, n:ast.IfExp):
        s.generic_visit(n)
        ok, v = static(n.test)
        return (n.body if v else n.orelse) if ok else n
      def visit_BoolOp(s, n:ast.BoolOp):
        s.generic_visit(n)
        is_and, vals = isinstance(n.op, ast.And), []
        for i,v in enumerate(n.values):
          ok, sv = static(v)
          # a static operand that short circuits ends the expression, only a literal can replace it
          if ok and bool(sv) != is_and:
            vals.append(ast.Constant(sv) if isinstance(sv, (bool, int, type(None))) else v)
            break
          # a static operand that doesn't short circuit is dropped, unless it's the value of the expression
          if ok and i != len(n.values)-1: continue
          vals.append(v)
        return vals[0] if len(vals) == 1 else ast.BoolOp(n.op, vals)
    return Fold().visit(tree)

  def specialize(self, op) -> Callable:
    # resolve the statements, then fold the expressions in what's left (on a copy), then resolve the statements that became static
    body = copy.deepcopy(self.stmts(self.fdef.body, op))
    body = self.stmts([cast(ast.stmt, self.exprs(st, op)) for st in body], op) or [ast.Pass()]
    name = f"{self.fxn.__name__}_{getattr(op, 'name', op)}"
    fdef = ast.FunctionDef(name, self.fdef.args, body, [], self.fdef.returns, self.fdef.type_comment, [])
    mod = ast.fix_missing_locations(ast.Module([ast.copy_location(fdef, self.fdef)], type_ignores=[]))
    namespace: dict = {}
    exec(compile(mod, self.fxn.__code__.co_filename, "exec"), self.globs, namespace)  # type: ignore[attr-defined] # pylint: disable=W0122
    return namespace[name]

class OpTable(dict):
  """op -> the function specialized on that op, built on first use"""
  def __init__(self, fxn:Callable, op_expr:str):
    super().__init__()
    self.spec = _OpSpecializer(fxn, op_expr)
  def __missing__(self, op) -> Callable:
    self[op] = ret = self.spec.specialize(op)
    return ret
