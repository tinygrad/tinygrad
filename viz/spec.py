from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class GraphRewriteMetadata:
  """Specifies metadata about a single call to graph_rewrite"""
  loc: Tuple[str, int]
  """File_path, Lineno"""
  code_line: str
  """The Python line calling graph_rewrite"""
  kernel_name: Optional[str]
  """The kernel instance calling graph_rewrite"""
  kernel_code: Optional[str]
  """The final code rendered by the kernel instance connected with this graph_rewrite"""
  upats: List[Tuple[Tuple[str, int], str]]
  """List of all the UPats that matched and returned a not none value."""

@dataclass(frozen=True)
class GraphRewriteDetails(GraphRewriteMetadata):
  """Full details about a single call to graph_rewrite"""
  graphs: List[Dict[int, Tuple[str, str, List[int], str, str]]]
  """Sink at every stage of graph_rewrite"""
  diffs: List[List[str]]
  """.diff style before and after of the rewritten UOp child"""
  changed_nodes: List[List[int]]
  """Nodes that changed at every stage of graph_rewrite"""
