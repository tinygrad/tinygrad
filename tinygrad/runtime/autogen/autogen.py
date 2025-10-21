import ctypes.util, importlib, importlib.metadata, os, pathlib, re, fcntl, functools
from tinygrad.helpers import unwrap
from itertools import takewhile

def fst(c): return next(c.get_children())
def last(c): return list(c.get_children())[-1]
def pread(f, count, offset):
  with open(f, "r") as f:
    f.seek(offset)
    return f.read(count)

def until(pred, f, x):
  while not pred(x): x = f(x)
  return x

rules = [(r'\s*\\\n\s*', ' '), (r'//.*', ''), (r'\b(\d+)[UuLl]+\b', r'\1'),
         (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
         (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
         (r'\((unsigned )?(char)\)', '')]

class Autogen:
  def __init__(self, name, dll, files, args=[], prelude=[]):
    self.name, self.dll, self.loaded, self.files, self.args, self.prelude = name, dll, False, files, args, prelude
    if not os.path.exists(pathlib.Path(__file__).parent / f"{self.name}.py"): self.gen()

  def __getattr__(self, nm):
    if not self.loaded: self._mod, self.loaded = importlib.import_module(f"tinygrad.runtime.autogen.{self.name}"), True
    return getattr(self._mod, nm)

  def render(self, c):
    from clang.cindex import CursorKind as CK
    match c.kind:
      case CK.TRANSLATION_UNIT: self.lines += [line for c in c.get_children() if (line:=self.render(c))]
      case CK.VAR_DECL:
        if len(list(c.get_children())) == 0:
          print(f"WARNING: libclang did not parse {c.spelling}")
          return None
        return f"{c.spelling} = {self.render(fst(c))}"
      case CK.FUNCTION_DECL:
        if len(list(c.get_children())) == 0 or len(list(last(c).get_children())) == 0:
          print(f"WARNING: libclang did not parse {c.spelling}")
          return None
        return f"{c.spelling} = lambda {','.join(a.spelling for a in c.get_arguments())}: {self.render(fst(last(c)))}"
      case CK.RETURN_STMT: return self.render(fst(c)) # FIXME: this wont work for static functions
      case CK.PAREN_EXPR: return "(" + self.render(fst(c)) + ")"
      case CK.UNARY_OPERATOR: return ''.join({'!':'not '}.get(t.spelling,t.spelling) for t in c.get_tokens())
      case CK.BINARY_OPERATOR: return self.render(next(children:=c.get_children())) + c.spelling + self.render(next(children))
      case CK.UNEXPOSED_EXPR: return self.render(next(c.get_children()))
      case CK.CSTYLE_CAST_EXPR: return f"{self.tname(c.type)}({self.render(next(c.get_children()))})"
      case CK.DECL_REF_EXPR: return c.spelling
      case CK.INTEGER_LITERAL: return next(c.get_tokens()).spelling.replace('U', '').replace('L', '')
      case CK.CHARACTER_LITERAL | CK.STRING_LITERAL: return next(c.get_tokens()).spelling
      case CK.CALL_EXPR: return f"{c.spelling}({', '.join(self.render(a) for a in c.get_arguments())})"
      case _: raise NotImplementedError(f"unsupported expression {c.kind} in render")

  def gen(self):
    from clang.cindex import Config, Index, CursorKind as CK, TranslationUnit as TU, TokenKind as ToK, PrintingPolicy as PP, PrintingPolicyProperty
    assert importlib.metadata.version('clang')[:2] == "20"
    if not Config.loaded: Config.set_library_file(ctypes.util.find_library("clang-20"))

    idx, self.lines = Index.create(), [f"import {', '.join(['ctypes'] + [i for i in ['ctypes.util'] if i in (self.dll or '')])}",
      "from tinygrad.runtime.autogen.helpers import *", *([f"dll = {self.dll}\n"] if self.dll else []), *self.prelude]
    self.types, self.macros, self.anoncnt = {}, set(), 0
    macros:list[str] = []
    for f in self.files() if callable(self.files) else self.files:
      tu = idx.parse(f, self.args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
      (pp:=PP.create(tu.cursor)).set_property(PrintingPolicyProperty.TerseOutput, 1)
      for c in tu.cursor.walk_preorder():
        if str(c.location.file) != f: continue
        match c.kind:
          case CK.FUNCTION_DECL:
            self.lines.append(f"# {c.pretty_printed(pp)}\ntry: ({c.spelling}:=dll.{c.spelling}).restype,{c.spelling}.argtypes = "
              f"{self.tname(c.result_type)},[{', '.join(self.tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
          case CK.STRUCT_DECL | CK.UNION_DECL | CK.TYPEDEF_DECL | CK.ENUM_DECL: self.tname(c.type)
          case CK.MACRO_DEFINITION if len(toks:=list(c.get_tokens())) > 1:
            if toks[1].spelling == '(' and toks[0].extent.end.column == toks[1].extent.start.column:
              it = iter(toks[1:])
              args = [t.spelling for t in takewhile(lambda t:t.spelling!=')', it) if t.kind == ToK.IDENTIFIER]
              macros += [f"{c.spelling} = lambda {','.join(args)}: {pread(f, toks[-1].extent.end.offset - (begin:=next(it).location.offset), begin)}"]
            else: macros += [f"{c.spelling} = {pread(f, toks[-1].extent.end.offset - (begin:=toks[1].location.offset), begin)}"]
    main, macros = '\n'.join(self.lines) + '\n', [functools.reduce(lambda s,r:re.sub(r[0], r[1], s), rules, m) for m in macros]
    while True:
      try:
        exec(main + '\n'.join(macros), {})
        break
      except Exception as e:
        macrono = unwrap(e.lineno if isinstance(e, SyntaxError) else unwrap(unwrap(e.__traceback__).tb_next).tb_lineno) - main.count('\n') - 1
        print(f"Skipping {macros[macrono].split()[0]}: {e}")
        del macros[macrono]
    with open(pathlib.Path(__file__).parent / f"{self.name}.py", "w") as f: f.write(main + '\n'.join(macros))
    importlib.invalidate_caches()

  def tname(self, t) -> str:
    from clang.cindex import CursorKind as CK, TypeKind as TK

    tmap = {TK.VOID:"None", TK.CHAR_U:"ctypes.c_ubyte", TK.UCHAR:"ctypes.c_ubyte", TK.CHAR_S:"ctypes.c_char", TK.SCHAR:"ctypes.c_char",
            **{getattr(TK, k):f"ctypes.c_{k.lower()}" for k in
            ["BOOL", "USHORT", "UINT", "ULONG", "ULONGLONG", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "FLOAT", "DOUBLE", "LONGDOUBLE"]}}
    pmap = {TK.VOID:"ctypes.c_void_p", TK.WCHAR:"ctypes.c_wchar_p", **{k:"ctypes.c_char_p" for k in [TK.UCHAR, TK.SCHAR, TK.CHAR_S, TK.CHAR_U]}}

    if t.kind in tmap: return tmap[t.kind]
    if t.spelling in self.types: return self.types[t.spelling]
    match t.kind:
      case TK.POINTER: return pmap[t.get_pointee().kind] if t.get_pointee().kind in pmap else f"ctypes.POINTER({self.tname(t.get_pointee())})"
      case TK.ELABORATED: return self.tname(t.get_named_type())
      case TK.TYPEDEF:
        self.types[t.spelling] = self.tname(t.get_canonical()) if t.spelling.startswith("__") else t.spelling
        self.lines.append(f"{t.spelling} = {self.tname(t.get_canonical())}")
        return self.types[t.spelling]
      case TK.RECORD:
        if (decl:=t.get_declaration()).is_anonymous():
          self.types[t.spelling] = f"_anon{'struct' if decl.kind == CK.STRUCT_DECL else 'union'}{self.anoncnt}"
          self.anoncnt += 1
        else: self.types[t.spelling] = t.spelling.replace(' ', '_')
        self.lines.append(f"class {self.types[t.spelling]}(ctypes.{'Structure' if decl.kind == CK.STRUCT_DECL else 'Union'}):\n" +
          (f"  _pack_ = {align}\n" if (align:=t.get_align()) > 0 else "") + "  _fields_ = [" +
          (",\n"+" "*14).join(f"('{f.spelling}', {self.tname(f.type)}{f', {f.get_bitfield_width()}' if f.is_bitfield() else ''})"
                              for f in t.get_fields()) + "]\n")
        return self.types[t.spelling]
      case TK.ENUM:
        if (decl:=t.get_declaration()).is_anonymous():
          self.types[t.spelling] = f"_anonenum{self.anoncnt}"
          self.anoncnt += 1
        else: self.types[t.spelling] = t.spelling.replace(' ', '_')
        self.lines.append(f"{self.types[t.spelling]} = CEnum({self.tname(decl.enum_type)})\n" +
          "\n".join(f"{e.spelling} = {self.types[t.spelling]}.define('{e.spelling}', {e.enum_value})" for e in decl.get_children()) + "\n")
        return self.types[t.spelling]
      case TK.FUNCTIONPROTO:
        return f"ctypes.CFUNCTYPE({self.tname(t.get_result())}{((', '+', '.join(self.tname(a) for a in ats)) if (ats:=t.argument_types()) else '')})"
      case TK.CONSTANTARRAY: return f"({self.tname(t.get_array_element_type())} * {t.get_array_size()})"
      case TK.INCOMPLETEARRAY: return f"({self.tname(t.get_array_element_type())} * {0})"
      case _: raise NotImplementedError(f"unsupported type {t.kind} at {t.location.file}:{t.location.line}:{t.location.column}")
