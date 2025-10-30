import ctypes.util, glob, importlib.metadata, itertools, re, functools, tarfile
from tinygrad.helpers import fetch, flatten, unwrap
from clang.cindex import Config, Index, CursorKind as CK, TranslationUnit as TU, LinkageKind as LK, TokenKind as ToK, TypeKind as TK
from clang.cindex import PrintingPolicy as PP, PrintingPolicyProperty as PPP, SourceRange

assert importlib.metadata.version('clang')[:2] == "20"
if not Config.loaded: Config.set_library_file(ctypes.util.find_library("clang-20"))

def fst(c): return next(c.get_children())
def last(c): return list(c.get_children())[-1]
def readext(f, fst, snd=None):
  with open(f, "r") as f:
    f.seek(start:=(fst.start.offset if isinstance(fst, SourceRange) else fst))
    return f.read((fst.end.offset if isinstance(fst, SourceRange) else snd)-start)

def until(pred, f, x):
  while not pred(x): x = f(x)
  return x

base_rules = [(r'\s*\\\n\s*', ' '), (r'\s*\n\s*', ' '), (r'//.*', ''), (r'/\*.*?\*/', ''), (r'\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1'),
              (r'\b0+(?=\d)', ''), (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
              (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
              (r'\((unsigned )?(char)\)', ''), (r'^.*[?;].*$', ''), (r'^.*\d+:\d+.*$', ''), (r'^.*\w##\w.*$', '')]

def gen(dll, files, args=[], prelude=[], rules=[], tarball=None, recsym=False, use_errno=False):
  files, args = files() if callable(files) else files, args() if callable(args) else args
  if tarball:
    # dangerous for arbitrary urls!
    with tarfile.open(fetch(tarball, gunzip=tarball.endswith("gz"))) as tf:
      tf.extractall("/tmp")
      base = f"/tmp/{tf.getnames()[0]}"
      files, args = [str(f).format(base) for f in files], [a.format(base) for a in args]
  files = flatten(glob.glob(p, recursive=True) if isinstance(p, str) and '*' in p else [p] for p in files)

  idx, lines = Index.create(), ["# mypy: ignore-errors", "import ctypes"+(', os' if any('os' in s for s in dll) else ''),
                                *(["from ctypes.util import find_library"] if any('find_library' in s for s in dll) else []),
                                "from tinygrad.helpers import CEnum, _IO, _IOW, _IOR, _IOWR", *prelude]
  if dll: lines += flatten(["def dll():",[[f"  try: return ctypes.CDLL({d}{', use_errno=True' if use_errno else ''})",'  except: pass'] for d in dll],
                            "  return None", "dll = dll()"])
  macros = []
  types:dict[str,str] = {}

  anoncnt = itertools.count().__next__
  def tname(t, suggested_name=None) -> str:
    nonlocal lines, types, anoncnt
    tmap = {TK.VOID:"None", TK.CHAR_U:"ctypes.c_ubyte", TK.UCHAR:"ctypes.c_ubyte", TK.CHAR_S:"ctypes.c_char", TK.SCHAR:"ctypes.c_char",
            **{getattr(TK, k):f"ctypes.c_{k.lower()}" for k in
            ["BOOL", "USHORT", "UINT", "ULONG", "ULONGLONG", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "FLOAT", "DOUBLE", "LONGDOUBLE"]}}

    if t.kind in tmap: return tmap[t.kind]
    if t.spelling in types: return types[t.spelling]
    if ((f:=t).kind in (fks:=(TK.FUNCTIONPROTO, TK.FUNCTIONNOPROTO))) or (t.kind == TK.POINTER and (f:=t.get_pointee()).kind in fks):
      return f"ctypes.CFUNCTYPE({tname(f.get_result())}{(', '+', '.join(map(tname, f.argument_types()))) if f.kind==TK.FUNCTIONPROTO else ''})"
    match t.kind:
      case TK.POINTER: return "ctypes.c_void_p" if t.get_pointee().kind == TK.VOID else f"ctypes.POINTER({tname(t.get_pointee())})"
      case TK.ELABORATED: return tname(t.get_named_type(), suggested_name)
      case TK.TYPEDEF if t.spelling == t.get_canonical().spelling: return tname(t.get_canonical())
      case TK.TYPEDEF:
        types[t.spelling] = tname(t.get_canonical()) if t.spelling.startswith("__") else t.spelling.replace('::', '_')
        lines.append(f"{t.spelling.replace('::', '_')} = {tname(t.get_canonical())}")
        return types[t.spelling]
      case TK.RECORD:
        if (decl:=t.get_declaration()).is_anonymous():
          types[t.spelling] = nm = suggested_name or (f"_anon{'struct' if decl.kind == CK.STRUCT_DECL else 'union'}{anoncnt()}")
        else: types[t.spelling] = nm = t.spelling.replace(' ', '_').replace('::', '_')
        lines.append(f"class {nm}(ctypes.{'Structure' if decl.kind==CK.STRUCT_DECL else 'Union'}): pass")
        aa, acnt = [], itertools.count().__next__
        ll=["  ("+((aa.append(fn:=f"'_{acnt()}'") or fn)+f", {tname(f.type)}" if f.is_anonymous_record_decl() else f"'{f.spelling}', "+
            tname(f.type, f'{nm}_{f.spelling}'))+(f',{f.get_bitfield_width()}' if f.is_bitfield() else '')+")," for f in t.get_fields()]
        lines.extend(([f"{nm}._anonymous_ = [{', '.join(aa)}]"] if aa else [])+[f"{nm}._fields_ = [",*ll,"]"] if ll else [f"{nm}._fields_ = []"])
        return nm
      case TK.ENUM:
        if (decl:=t.get_declaration()).is_anonymous(): types[t.spelling] = suggested_name or f"_anonenum{anoncnt()}"
        else: types[t.spelling] = t.spelling.replace(' ', '_').replace('::', '_')
        lines.append(f"{types[t.spelling]} = CEnum({tname(decl.enum_type)})\n" +
          "\n".join(f"{e.spelling} = {types[t.spelling]}.define('{e.spelling}', {e.enum_value})" for e in decl.get_children()
                    if e.kind == CK.ENUM_CONSTANT_DECL) + "\n")
        return types[t.spelling]
      case TK.CONSTANTARRAY: return f"({tname(t.get_array_element_type())} * {t.get_array_size()})"
      case TK.INCOMPLETEARRAY: return f"({tname(t.get_array_element_type())} * {0})"
      case _: raise NotImplementedError(f"unsupported type {t.kind}")

  for f in files:
    tu = idx.parse(f, args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
    (pp:=PP.create(tu.cursor)).set_property(PPP.TerseOutput, 1)
    for c in tu.cursor.walk_preorder():
      if str(c.location.file) != str(f) and (not recsym or c.kind not in (CK.FUNCTION_DECL,)): continue
      try:
        match c.kind:
          case CK.FUNCTION_DECL if c.linkage == LK.EXTERNAL and dll:
            lines.append(f"# {c.pretty_printed(pp)}\ntry: ({c.spelling}:=dll.{c.spelling}).restype, {c.spelling}.argtypes = "
              f"{tname(c.result_type)}, [{', '.join(tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
          case CK.STRUCT_DECL | CK.UNION_DECL | CK.TYPEDEF_DECL | CK.ENUM_DECL: tname(c.type)
          case CK.MACRO_DEFINITION if len(toks:=list(c.get_tokens())) > 1:
            if toks[1].spelling == '(' and toks[0].extent.end.column == toks[1].extent.start.column:
              it = iter(toks[1:])
              _args = [t.spelling for t in itertools.takewhile(lambda t:t.spelling!=')', it) if t.kind == ToK.IDENTIFIER]
              if len(body:=list(it)) == 0: continue
              macros += [f"{c.spelling} = lambda {','.join(_args)}: {readext(f, body[0].location.offset, toks[-1].extent.end.offset)}"]
            else: macros += [f"{c.spelling} = {readext(f, toks[1].location.offset, toks[-1].extent.end.offset)}"]
          case CK.VAR_DECL if c.linkage == LK.INTERNAL:
            if (c.type.kind == TK.CONSTANTARRAY and c.type.get_array_element_type().kind in (TK.INT,TK.UINT) and
                (init:=last(c)).kind == CK.INIT_LIST_EXPR and all(re.match(r"\[.*\].*=", readext(f, c.extent)) for c in init.get_children())):
              macros += [f"{c.spelling} = {{{','.join(f'{readext(f, next(it:=c.get_children()).extent)}:{readext(f, next(it).extent)}' for c in init.get_children())}}}"]
            else: macros += [f"{c.spelling} = {tname(c.type)}({readext(f, last(c).extent)})"]
      except Exception as e: raise Exception(f"parsing failed at {c.location.file}:{c.location.line}") from e
  main, macros = '\n'.join(lines) + '\n', [r for m in macros if (r:=functools.reduce(lambda s,r:re.sub(r[0], r[1], s), rules + base_rules, m))]
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
  return main + '\n'.join(macros)
