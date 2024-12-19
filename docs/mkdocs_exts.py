import griffe

class StateAcceptFilenameDecorator(griffe.Extension):
  def on_function_instance(self, *, func: griffe.Function, **_) -> None:
    if any(d.callable_path == "tinygrad.nn.state.accept_filename" for d in func.decorators):
      def name(n: str): return griffe.ExprName(n, func.parent)
      for p in func.parameters:
        p.annotation = griffe.ExprSubscript(name("Union"), griffe.ExprTuple([name("Tensor"), name("str"), name("pathlib.Path")], implicit=True))
