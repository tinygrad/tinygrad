import sys, json
from tinygrad.uop.spec import eval_pyrender

ast = None
for data in sys.stdin:
  try: data = json.loads(data)
  except: continue
  if "ast = " in data.get("value", ""):
    ast = eval_pyrender(data["value"])
assert ast is not None
print(type(ast))
print(ast.pyrender())
