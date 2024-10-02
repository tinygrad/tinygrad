from dataclasses import dataclass
from typing import List, Optional, Tuple
from tinygrad.ops import UOp

@dataclass(frozen=True)
class GraphRewriteMetadata:
  """Specifies metadata about a single call to graph_rewrite"""
  loc: Tuple[str, int]
  """File_path, Lineno"""
  code_line: str
  """The python line calling graph_rewrite"""
  kernel_name: Optional[str]
  """The kernel instance calling graph_rewrite"""
  kernel_code: Optional[str]
  """The final code rendered by the kernel instance connected with this graph_rewrite"""
  rewrite_count: int
  """Total number of rewrites on the sink"""

@dataclass(frozen=True)
class GraphRewriteFullInfo:
  """
  Full details about a single call to graph_rewrite
  NOTE: this is slower to get because it reconstructs the entire graph after every rewrite
  """
  graphs: List[UOp]
  """Snapshot of the SINK at every stage of graph_rewrite"""
  upats: List[Tuple[Tuple[str, int], str]]
  """List of all the UPats that matched and returned a not none value."""
  diffs: List[str]
  """.diff style before and after of the rewritten UOp child"""
  metadata: GraphRewriteMetadata
