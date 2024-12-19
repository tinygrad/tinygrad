import griffe

class StateAcceptFilenameDecorator(griffe.Extension):
  def on_function_instance(self, *, func: griffe.Function, **kwargs) -> None:
    def name(n: str): return griffe.ExprName(n, func.parent)
    for decorator in func.decorators:
      if decorator.callable_path == "tinygrad.nn.state.accept_filename":
        for p in func.parameters:
          p.annotation = griffe.ExprSubscript(name("Union"), griffe.ExprTuple([name("Tensor"), name("str"), name("pathlib.Path")], implicit=True))