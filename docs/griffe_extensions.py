import ast, griffe

# autogen modules are never documented and have tens of thousands of top level statements, which griffe is quadratic in
class SkipAutogen(griffe.Extension):
  def on_module_node(self, *, node, agent, **kwargs):
    if isinstance(node, ast.Module) and "runtime/autogen" in str(agent.filepath): node.body = []
