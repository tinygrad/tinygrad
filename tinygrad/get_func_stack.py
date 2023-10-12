import inspect
# get the list of all function calls taht leads to the current function from the stack
def generate_stack():
  for i in list(reversed(inspect.stack())): print(f"Function = {i.function} called from {i.filename} at line number ={i.lineno}")