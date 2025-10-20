import ctypes.util, importlib, importlib.metadata, os, pathlib, re
from itertools import takewhile

def CEnum(typ):
  class _CEnum(typ):
    _val_to_name_ = {}

    @classmethod
    def from_param(cls, val): return val if isinstance(val, cls) else cls(val)
    @classmethod
    def name(cls, val): return cls._val_to_name_.get(val.value if isinstance(val, cls) else val)
    @classmethod
    def define(cls, name, val):
      cls._val_to_name_[val] = name
      return cls(val)

    def __eq__(self, other): return self.value == other
    def __repr__(self): return self.name(self) if self.value in self.__class__._val_to_name_ else str(self.value)

  return _CEnum

def fst(c): return next(c.get_children())

class Autogen:
  def __init__(self, name, dll, files, args=[]):
    self.name, self.dll, self.files, self.args, self.types, self.macros, self.anoncnt = name, dll, files, args, {}, set(), 0
    self._mod = self.gen()

  def __getattr__(self, nm): return getattr(self._mod, nm)

  def render(self, c):
    from clang.cindex import CursorKind as CK
    match c.kind:
      case CK.RETURN_STMT: return f"return {self.render(next(c.get_children()))}"
      case CK.PAREN_EXPR: return "(" + self.render(next(c.get_children())) + ")"
      case CK.UNARY_OPERATOR: return ''.join(t.spelling for t in c.get_tokens())
      case CK.BINARY_OPERATOR: return self.render(next(children:=c.get_children())) + c.spelling + self.render(next(children))
      case CK.UNEXPOSED_EXPR: return self.render(next(c.get_children()))
      case CK.CSTYLE_CAST_EXPR: return f"{self.tname(c.type)}({self.render(next(c.get_children()))})"
      case CK.DECL_REF_EXPR: return c.spelling
      case CK.INTEGER_LITERAL: return next(c.get_tokens()).spelling.replace('U', '').replace('L', '')
      case CK.CHARACTER_LITERAL | CK.STRING_LITERAL: return next(c.get_tokens()).spelling
      case CK.CALL_EXPR: return f"{c.spelling}({', '.join(self.render(a) for a in c.get_arguments())})"
      case _: raise NotImplementedError(f"unsupported expression {c.kind} in render")

  def gen(self):
    if not os.path.exists(path:=(pathlib.Path(__file__).parent / f"{self.name}.py")):
      from clang.cindex import Config, Index, CursorKind as CK, TranslationUnit as TU, TokenKind as ToK, PrintingPolicy as PP, PrintingPolicyProperty as PPP
      assert importlib.metadata.version('clang')[:2] == "20"
      if not Config.loaded: Config.set_library_file(ctypes.util.find_library("clang-20"))

      idx, self.lines = Index.create(), [f"import {', '.join(['ctypes'] + [i for i in ['ctypes.util'] if i in self.dll])}\ndll = {self.dll}\n",
                                         "from tinygrad.runtime.autogen.autogen import CEnum"]
      for f in self.files() if callable(self.files) else self.files:
        macros:list[tuple[str,tuple[str,...]]] = []
        tu = idx.parse(f, self.args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
        (pp:=PP.create(tu.cursor)).set_property(PPP.TerseOutput, 1)
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
                body = ' '.join(t.spelling for t in it)
                mtu = idx.parse("tmp.c", unsaved_files=[("tmp.c", f"{';'.join(args)};_x = {body};")])
                if len(list((decl:=list(mtu.cursor.get_children())[-1]).get_children())) == 0: continue
                try: macros += [(f"def {c.spelling}({', '.join(a for a in args)}): return " + self.render(fst(decl)), ())]
                except Exception as e: raise Exception(f"{c.location}") from e
              elif len(toks) == 2:
                macros += [(c.spelling+" = "+re.sub(r'\b(\d+)(L|U)',r'\1',(t:=toks[1]).spelling), (t.spelling,) if t.kind == ToK.IDENTIFIER else ())]
              else:
                mtu = idx.parse("tmp.c", unsaved_files=[("tmp.c", f"_x = {' '.join(t.spelling for t in toks[1:])};")])
                if len(list((decl:=fst(mtu.cursor)).get_children())) == 0: continue
                try: macros += [(f"{c.spelling} = {self.render(fst(decl))}", ())]
                except Exception as e: raise Exception(f"{c.location}") from e
        self.lines += [m[0] for m in macros if all([dep in '\n'.join(self.lines) for dep in m[1]])]
      with open(path, "w") as f: f.write('\n'.join(self.lines))
      importlib.invalidate_caches()
    return importlib.import_module(f"tinygrad.runtime.autogen.{self.name}")

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
        self.types[t.spelling] = t.spelling
        self.lines.append(f"{t.spelling} = {self.tname(t.get_canonical())}")
        return t.spelling
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
