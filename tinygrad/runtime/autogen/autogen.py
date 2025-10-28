import ctypes.util, glob, importlib, importlib.metadata, itertools, os, pathlib, re, functools, tarfile
from tinygrad.helpers import fetch, flatten, unwrap

def fst(c): return next(c.get_children())
def last(c): return list(c.get_children())[-1]
def pread(f, count, offset):
  with open(f, "r") as f:
    f.seek(offset)
    return f.read(count)

def until(pred, f, x):
  while not pred(x): x = f(x)
  return x

base_rules = [(r'\s*\\\n\s*', ' '), (r'//.*', ''), (r'/\*.*?\*/', ''), (r'\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1'), (r'\b0+(?=\d)', ''),
         (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
         (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
         (r'\((unsigned )?(char)\)', ''), (r'^.*[?;].*$', ''), (r'^.*\d+:\d+.*$', ''), (r'^.*\w##\w.*$', '')]

class Autogen:
  def __init__(self, name, dll, files, args=[], prelude=[], rules=[], tarball=None, recsym=False):
    self.name, self.dll, self.loaded, self.files, self.args, self.prelude, self.tarball = name, dll, False, files, args, prelude, tarball
    self.rules, self.recsym = rules + base_rules, recsym
    if not os.path.exists(pathlib.Path(__file__).parent / f"{self.name}.py"): self.gen()

  @functools.cached_property
  def _mod(self): return importlib.import_module(f"tinygrad.runtime.autogen.{self.name}")
  def __getattr__(self, nm): return getattr(self._mod, nm)

  def gen(self):
    from clang.cindex import Config, Index, CursorKind as CK, TranslationUnit as TU, LinkageKind as LK, TokenKind as ToK, PrintingPolicy as PP
    from clang.cindex import PrintingPolicyProperty as PPP
    assert importlib.metadata.version('clang')[:2] == "20"
    if not Config.loaded: Config.set_library_file(ctypes.util.find_library("clang-20"))

    self.files, self.args = self.files() if callable(self.files) else self.files, self.args() if callable(self.args) else self.args
    if self.tarball:
      # dangerous for arbitrary urls!
      with tarfile.open(fetch(self.tarball, gunzip=self.tarball.endswith("gz"))) as tf:
        tf.extractall("/tmp")
        base = f"/tmp/{tf.getnames()[0]}"
        self.files, self.args = [str(f).format(base) for f in self.files], [a.format(base) for a in self.args]
    self.files = flatten(glob.glob(p, recursive=True) if isinstance(p, str) and '*' in p else [p] for p in self.files)

    idx, self.lines = Index.create(), ["# mypy: ignore-errors", "import ctypes"+(', ctypes.util' if 'ctypes.util' in (self.dll or '') else ''),
      "from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR", *self.prelude, *([f"dll = {self.dll}\n"] if self.dll else [])]
    self.types, self.macros, self.anoncnt = {}, set(), itertools.count().__next__
    macros:list[str] = []
    for f in self.files:
      tu = idx.parse(f, self.args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
      (pp:=PP.create(tu.cursor)).set_property(PPP.TerseOutput, 1)
      for c in tu.cursor.walk_preorder():
        if str(c.location.file) != str(f) and (not self.recsym or c.kind not in (CK.FUNCTION_DECL,)): continue
        match c.kind:
          case CK.FUNCTION_DECL if c.linkage == LK.EXTERNAL and self.dll is not None:
            self.lines.append(f"# {c.pretty_printed(pp)}\ntry: ({c.spelling}:=dll.{c.spelling}).restype, {c.spelling}.argtypes = "
              f"{self.tname(c.result_type)}, [{', '.join(self.tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
          case CK.STRUCT_DECL | CK.UNION_DECL | CK.TYPEDEF_DECL | CK.ENUM_DECL: self.tname(c.type)
          case CK.MACRO_DEFINITION if len(toks:=list(c.get_tokens())) > 1:
            if toks[1].spelling == '(' and toks[0].extent.end.column == toks[1].extent.start.column:
              it = iter(toks[1:])
              args = [t.spelling for t in itertools.takewhile(lambda t:t.spelling!=')', it) if t.kind == ToK.IDENTIFIER]
              if len(body:=list(it)) == 0: continue
              macros += [f"{c.spelling} = lambda {','.join(args)}: {pread(f, toks[-1].extent.end.offset - (begin:=body[0].location.offset), begin)}"]
            else: macros += [f"{c.spelling} = {pread(f, toks[-1].extent.end.offset - (begin:=toks[1].location.offset), begin)}"]
    main, macros = '\n'.join(self.lines) + '\n', [r for m in macros if (r:=functools.reduce(lambda s,r:re.sub(r[0], r[1], s), self.rules, m))]
    while True:
      try:
        exec(main + '\n'.join(macros), {})
        break
      except (SyntaxError, NameError, TypeError) as e:
        macrono = unwrap(e.lineno if isinstance(e, SyntaxError) else unwrap(unwrap(e.__traceback__).tb_next).tb_lineno) - main.count('\n') - 1
        assert macrono >= 0 and macrono < len(macros)
        print(f"Skipping {macros[macrono]}: {e}")
        del macros[macrono]
      except FileNotFoundError: break
    with open(pathlib.Path(__file__).parent / f"{self.name}.py", "w") as f: f.write(main + '\n'.join(macros))
    importlib.invalidate_caches()

  def tname(self, t, suggested_name=None) -> str:
    from clang.cindex import CursorKind as CK, TypeKind as TK

    tmap = {TK.VOID:"None", TK.CHAR_U:"ctypes.c_ubyte", TK.UCHAR:"ctypes.c_ubyte", TK.CHAR_S:"ctypes.c_char", TK.SCHAR:"ctypes.c_char",
            **{getattr(TK, k):f"ctypes.c_{k.lower()}" for k in
            ["BOOL", "USHORT", "UINT", "ULONG", "ULONGLONG", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "FLOAT", "DOUBLE", "LONGDOUBLE"]}}

    if t.kind in tmap: return tmap[t.kind]
    if t.spelling in self.types: return self.types[t.spelling]
    match t.kind:
      case TK.POINTER if (f:=t.get_pointee()).kind == TK.FUNCTIONPROTO:
        return f"ctypes.CFUNCTYPE({self.tname(f.get_result())}{((', '+', '.join(self.tname(a) for a in ats)) if (ats:=f.argument_types()) else '')})"
      case TK.POINTER: return "ctypes.c_void_p" if t.get_pointee().kind == TK.VOID else f"ctypes.POINTER({self.tname(t.get_pointee())})"
      case TK.ELABORATED: return self.tname(t.get_named_type(), suggested_name)
      case TK.TYPEDEF if t.spelling == t.get_canonical().spelling: return self.tname(t.get_canonical())
      case TK.TYPEDEF:
        self.types[t.spelling] = self.tname(t.get_canonical()) if t.spelling.startswith("__") else t.spelling
        self.lines.append(f"{t.spelling} = {self.tname(t.get_canonical())}")
        return self.types[t.spelling]
      case TK.RECORD:
        if (decl:=t.get_declaration()).is_anonymous():
          self.types[t.spelling] = nm = suggested_name or (f"_anon{'struct' if decl.kind == CK.STRUCT_DECL else 'union'}{self.anoncnt()}")
        else: self.types[t.spelling] = nm = t.spelling.replace(' ', '_')
        self.lines.append(f"class {nm}(ctypes.{'Structure' if decl.kind==CK.STRUCT_DECL else 'Union'}): pass")
        aa, acnt = [], itertools.count().__next__
        ll=["  ("+((aa.append(fn:=f"'_{acnt()}'") or fn)+f", {self.tname(f.type)}" if f.is_anonymous_record_decl() else f"'{f.spelling}', "+
            self.tname(f.type, f'{nm}_{f.spelling}'))+(f',{f.get_bitfield_width()}' if f.is_bitfield() else '')+")," for f in t.get_fields()]
        self.lines.extend((aa if acnt() != 1 else [])+[f"{nm}._fields_ = [",*ll,"]"] if ll else [f"{nm}._fields_ = []"])
        return nm
      case TK.ENUM:
        if (decl:=t.get_declaration()).is_anonymous(): self.types[t.spelling] = suggested_name or f"_anonenum{self.anoncnt()}"
        else: self.types[t.spelling] = t.spelling.replace(' ', '_')
        self.lines.append(f"{self.types[t.spelling]} = CEnum({self.tname(decl.enum_type)})\n" +
          "\n".join(f"{e.spelling} = {self.types[t.spelling]}.define('{e.spelling}', {e.enum_value})" for e in decl.get_children()) + "\n")
        return self.types[t.spelling]
      case TK.CONSTANTARRAY: return f"({self.tname(t.get_array_element_type())} * {t.get_array_size()})"
      case TK.INCOMPLETEARRAY: return f"({self.tname(t.get_array_element_type())} * {0})"
      case _: raise NotImplementedError(f"unsupported type {t.kind} at {t.location.file}:{t.location.line}:{t.location.column}")
