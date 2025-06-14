from typing import Any, Callable
import functools
from dataclasses import dataclass
from tinygrad.helpers import QUANTIZE, DEVECTORIZE, TRANSCENDENTAL
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp
from tinygrad.renderer import Renderer

# import all pattern matchers here
from tinygrad.codegen.lowerer import pm_quant, pm_lowerer, get_index
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing
from tinygrad.codegen.expander import migrate_indexing, expander
from tinygrad.codegen.devectorizer import load_store_folding, load_store_indexing, devectorize, \
  pm_reduce, ReduceContext, correct_load_store, pm_render, get_late_rewrite_patterns
from tinygrad.codegen.linearize import block_create, pm_blockend_merge, block_merge, pm_finalize, BlockContext

@dataclass
class RewriteStep:
  pm: PatternMatcher
  ctx: Callable[[UOp], Any]|None = None
  name: str|None = None
  bottom_up: bool = False
  def __call__(self, sink:UOp):
    return graph_rewrite(sink, self.pm, ctx=self.ctx(sink) if self.ctx is not None else None, name=self.name, bottom_up=self.bottom_up)

def apply_rewrites(sink:UOp, rewrites:list[RewriteStep]): return functools.reduce(lambda x,f: f(x), rewrites, sink)

rewrites_for_linearizer = [
  RewriteStep(block_create, ctx=BlockContext.from_sink, name="Linearizer: Create Blocks", bottom_up=True),
  RewriteStep(pm_blockend_merge, name="Linearizer: Merge Blockends"),
  RewriteStep(block_merge, name="Linearizer: Merge Blocks"),
  RewriteStep(pm_finalize, name="Linearizer: Finalize")]

def get_rewrites_for_renderer(opts:Renderer, linearizer:bool=True) -> list[RewriteStep]:
  # cache with the values of the context vars
  return _get_rewrites_for_renderer(opts, linearizer, QUANTIZE.value, DEVECTORIZE.value, TRANSCENDENTAL.value)

@functools.cache
def _get_rewrites_for_renderer(opts:Renderer, linearizer:bool, _QUANTIZE, _DEVECTORIZE, _TRANSCENDENTAL) -> list[RewriteStep]:
  # ** lowerer (rewrite_shapetracker_with_index) **
  ret: list[RewriteStep] = []
  if _QUANTIZE and opts.device in {"CPU", "DSP"}: ret.append(RewriteStep(pm_quant, name="quantize"))
  ret.append(RewriteStep(pm_lowerer, lambda ast: get_index(ast, opts), name="lowerer"))

  # ** expander (expand_rewrite) **
  ret.append(RewriteStep(sym+migrate_indexing, name="initial symbolic"))

  # expand
  ret.append(RewriteStep(sym+expander, name="expander"))

  # ** devectorizer (full_graph_rewrite) **
  # remove reduce
  ret.append(RewriteStep(pm_reduce+gep_pushing, lambda _: ReduceContext(), name="remove_reduce"))

  # devectorize (TODO: does this need opts?)
  if _DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
  elif _DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
  else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
  ret.append(RewriteStep(pm_devectorize, lambda _: opts, name="devectorize"))

  supported_ops = tuple(opts.code_for_op.keys())
  extra_matcher = opts.extra_matcher if opts.extra_matcher is not None else PatternMatcher([])

  # optional pre matcher
  if opts.pre_matcher is not None: ret.append(RewriteStep(opts.pre_matcher, name="pre_matcher"))

  # final rules for the renderer (without sym)
  pm_final_rewrite = symbolic_simple+get_late_rewrite_patterns(supported_ops, _TRANSCENDENTAL>=2)+pm_render+extra_matcher
  ret.append(RewriteStep(pm_final_rewrite, lambda _: opts, name="final rewrite"))

  # return the list (with optional linearizer)
  return ret + (rewrites_for_linearizer if linearizer else [])

def full_rewrite_to_sink(sink:UOp, opts:Renderer|None=None, linearizer:bool=False) -> UOp:
  return apply_rewrites(sink, get_rewrites_for_renderer(opts if opts is not None else Renderer(), linearizer))
def full_rewrite(sink:UOp, opts:Renderer|None=None) -> list[UOp]: return list(full_rewrite_to_sink(sink, opts, linearizer=True).arg.lst)
