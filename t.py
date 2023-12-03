import ast
my_str = open("./all").read()
splts = my_str.splitlines()
fns = []
failed_tests = []
for s in splts:
  if 'TestHalfOps' in s:
      fn = s.split(" ")[-1].replace("testMethod=", "").replace(">", "")
      fns.append(fn)
  else:
    if len(fns) == 0: continue
    if 'not implemented for' in s:
      failed_tests.append(fns[-1])
test_ops = open("./test/test_ops.py").read()
def ast_to_code(node): return ast.unparse(node) if hasattr(ast, 'unparse') else None
parsed_code = ast.parse(test_ops)
test_ops_methods = []
test_float_ops = None
class MyVisitor(ast.NodeTransformer):
  def visit_ClassDef(self, node):
    global test_float_ops
    if node.name == "TestOps":
      methods_to_move = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in failed_tests]
      test_ops_methods.extend(methods_to_move)
      node.body = [item for item in node.body if item not in methods_to_move]
    elif node.name == "TestFloatOps": test_float_ops = node
    return node
visitor = MyVisitor()
modified_ast = visitor.visit(parsed_code)
if test_float_ops is not None: test_float_ops.body.extend(test_ops_methods)
with open("./test/test_ops.py", "w") as f: f.write(ast_to_code(modified_ast))
