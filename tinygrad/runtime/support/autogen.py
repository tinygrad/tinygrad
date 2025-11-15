import ctypes.util, importlib.metadata, itertools, re, functools, os
from tinygrad.helpers import flatten, unwrap, fromimport

assert importlib.metadata.version('clang')[:2] == "20", 'clang version 20 required, pip install "clang==20.1.0"'
from clang.cindex import Config, Index, Cursor, Type, CursorKind as CK, TranslationUnit as TU, LinkageKind as LK, TokenKind as ToK, TypeKind as TK
from clang.cindex import PrintingPolicy as PP, PrintingPolicyProperty as PPP, SourceRange

libclang = functools.partial(fromimport, "tinygrad.runtime.autogen.libclang") # we can't actually import this, because then we can't generate it

if not Config.loaded: Config.set_library_file(os.getenv("LIBCLANG_PATH", ctypes.util.find_library("clang-20")))

def fst(c): return next(c.get_children())
def last(c): return list(c.get_children())[-1]
def readext(f, fst, snd=None):
  with open(f, "r") as f:
    f.seek(start:=(fst.start.offset if isinstance(fst, SourceRange) else fst))
    return f.read((fst.end.offset if isinstance(fst, SourceRange) else snd)-start)
def attrs(c): return list(filter(lambda k: (v:=k.value) >= 400 and v < 500, map(lambda c: c.kind, c.get_children())))

def protocols(t): yield from (Cursor.from_result(libclang("clang_Type_getObjCProtocolDecl")(t, i), t)
                              for i in range(libclang("clang_Type_getNumObjCProtocolRefs")(t)))
def basetype(t): return Type.from_result(libclang("clang_Type_getObjCObjectBaseType")(t), (t,))

base_rules = [(r'\s*\\\n\s*', ' '), (r'\s*\n\s*', ' '), (r'//.*', ''), (r'/\*.*?\*/', ''), (r'\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1'),
              (r'\b0+(?=\d)', ''), (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
              (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
              (r'\((unsigned )?(char|uint64_t)\)', ''), (r'^.*\d+:\d+.*$', ''), (r'^.*\w##\w.*$', '')]

ints = (TK.INT, TK.UINT, TK.LONG, TK.ULONG, TK.LONGLONG, TK.ULONGLONG)
specs = (CK.OBJC_SUPER_CLASS_REF,)
# https://clang.llvm.org/docs/AutomaticReferenceCounting.html#arc-method-families
arc_families = ['alloc', 'copy', 'mutableCopy', 'new']

def gen(dll, files, args=[], prolog=[], rules=[], epilog=[], recsym=False, use_errno=False, anon_names={}, types={}, parse_macros=True):
  macros, lines, anoncnt, types, objc = [], [], itertools.count().__next__, {k:(v,True) for k,v in types.items()}, False
  def tname(t, suggested_name=None, typedef=None) -> str:
    suggested_name = anon_names.get(f"{(decl:=t.get_declaration()).location.file}:{decl.location.line}", suggested_name)
    nonlocal lines, types, anoncnt, objc
    tmap = {TK.VOID:"None", TK.CHAR_U:"ctypes.c_ubyte", TK.UCHAR:"ctypes.c_ubyte", TK.CHAR_S:"ctypes.c_char", TK.SCHAR:"ctypes.c_char",
            **{getattr(TK, k):f"ctypes.c_{k.lower()}" for k in ["BOOL", "WCHAR", "FLOAT", "DOUBLE", "LONGDOUBLE"]},
            **{getattr(TK, k):f"ctypes.c_{'u' if 'U' in k else ''}int{sz}" for sz,k in
               [(16, "USHORT"), (16, "SHORT"), (32, "UINT"), (32, "INT"), (64, "ULONG"), (64, "LONG"), (64, "ULONGLONG"), (64, "LONGLONG")]}}

    if t.kind in tmap: return tmap[t.kind]
    if t.spelling in types and types[t.spelling][1]: return types[t.spelling][0]
    if ((f:=t).kind in (fks:=(TK.FUNCTIONPROTO, TK.FUNCTIONNOPROTO))) or (t.kind == TK.POINTER and (f:=t.get_pointee()).kind in fks):
      return f"ctypes.CFUNCTYPE({tname(f.get_result())}{(', '+', '.join(map(tname, f.argument_types()))) if f.kind==TK.FUNCTIONPROTO else ''})"
    match t.kind:
      case TK.POINTER: return "ctypes.c_void_p" if (ptr:=t.get_pointee()).kind == TK.VOID else f"ctypes.POINTER({tname(ptr)})"
      case TK.OBJCOBJECTPOINTER: return tname(t.get_pointee()) # TODO: this seems wrong
      case TK.ELABORATED: return tname(t.get_named_type(), suggested_name)
      case TK.TYPEDEF if t.spelling == t.get_canonical().spelling: return tname(t.get_canonical())
      case TK.TYPEDEF:
        defined, nm = (canon:=t.get_canonical()).spelling in types, tname(canon, typedef=t.spelling.replace('::', '_'))
        types[t.spelling] = nm if t.spelling.startswith("__") else t.spelling.replace('::', '_'), True
        # RECORDs need to handle typedefs specially to allow for self-reference
        if canon.kind != TK.RECORD or defined: lines.append(f"{t.spelling.replace('::', '_')} = {nm}")
        return types[t.spelling][0]
      case TK.RECORD:
        # TODO: packed unions
        # check for forward declaration
        if t.spelling in types: types[t.spelling] = (nm:=types[t.spelling][0]), len(list(t.get_fields())) != 0
        else:
          if decl.is_anonymous():
            types[t.spelling] = (nm:=(suggested_name or (f"_anon{'struct' if decl.kind == CK.STRUCT_DECL else 'union'}{anoncnt()}")), True)
          else: types[t.spelling] = (nm:=t.spelling.replace(' ', '_').replace('::', '_')), len(list(t.get_fields())) != 0
          lines.append(f"class {nm}({'Struct' if decl.kind==CK.STRUCT_DECL else 'ctypes.Union'}): pass")
          if typedef: lines.append(f"{typedef} = {nm}")
        if (is_packed:=(CK.PACKED_ATTR in attrs(decl)) or ((N:=t.get_align()) != max([f.type.get_align() for f in t.get_fields()], default=N))):
          if t.get_align() != 1:
            print(f"WARNING: ignoring alignment={t.get_align()} on {t.spelling}")
            is_packed = False
        acnt = itertools.count().__next__
        ll=["  ("+((fn:=f"'_{acnt()}'")+f", {tname(f.type, nm+fn[1:-1])}" if f.is_anonymous_record_decl() else f"'{f.spelling}', "+
            tname(f.type, f'{nm}_{f.spelling}'))+(f',{f.get_bitfield_width()}' if f.is_bitfield() else '')+")," for f in t.get_fields()]
        lines.extend(([f"{nm}._anonymous_ = ["+", ".join(f"'_{i}'" for i in range(n))+"]"] if (n:=acnt()) else [])+
                     ([f"{nm}._packed_ = True"] * is_packed)+([f"{nm}._fields_ = [",*ll,"]"] if ll else []))
        return nm
      case TK.ENUM:
        # TODO: C++ and GNU C have forward declared enums
        if decl.is_anonymous(): types[t.spelling] = suggested_name or f"_anonenum{anoncnt()}", True
        else: types[t.spelling] = t.spelling.replace(' ', '_').replace('::', '_'), True
        lines.append(f"{types[t.spelling][0]} = CEnum({tname(decl.enum_type)})\n" +
                     "\n".join(f"{e.spelling} = {types[t.spelling][0]}.define('{e.spelling}', {e.enum_value})" for e in decl.get_children()
                     if e.kind == CK.ENUM_CONSTANT_DECL) + "\n")
        return types[t.spelling][0]
      case TK.CONSTANTARRAY:
        return f"({tname(t.get_array_element_type(), suggested_name.rstrip('s') if suggested_name else None)} * {t.get_array_size()})"
      case TK.INCOMPLETEARRAY: return f"({tname(t.get_array_element_type(), suggested_name.rstrip('s') if suggested_name else None)} * 0)"
      case TK.OBJCINTERFACE:
        is_defn = bool([f.kind for f in decl.get_children() if f.kind in (CK.OBJC_INSTANCE_METHOD_DECL, CK.OBJC_CLASS_METHOD_DECL)])
        if (nm:=t.spelling) not in types: lines.append(f"class {nm}(objc.Spec): pass")
        types[nm] = nm, is_defn
        if is_defn:
          ims, cms = parse_objc_spec(decl, t.spelling, CK.OBJC_INSTANCE_METHOD_DECL), parse_objc_spec(decl, t.spelling, CK.OBJC_CLASS_METHOD_DECL)
          lines.extend([*([f"{nm}._bases_ = [{', '.join(bs)}]"] if (bs:=[tname(b.type) for b in decl.get_children() if b.kind in specs]) else []),
                        *([f"{nm}._methods_ = [", *ims, ']'] if ims else []), *([f"{nm}._classmethods_ = [", *cms, ']'] if cms else [])])
        return nm
      case TK.OBJCSEL: return "objc.id_"
      case TK.OBJCID: return (objc:=True, "objc.id_")[1]
      case TK.OBJCOBJECT:
        if basetype(t).kind != TK.OBJCID: raise NotImplementedError(f"generics unsupported: {t.spelling}")
        ps = [proto(p) for p in protocols(t)]
        if len(ps) == 0:
          types[t.spelling] = "objc.id_", True
          return "objc.id_"
        if len(ps) == 1:
          types[t.spelling] = ps[0], True
          return ps[0]
        types[t.spelling] = (nm:=f"_anondynamic{anoncnt()}"), True
        lines.append(f"class {nm}({', '.join(p for p in ps)}): pass # {t.spelling}")
        return nm
      case _: raise NotImplementedError(f"unsupported type {t.kind}")

  # parses an objc @interface or @protocol, returning a list of declerations that objc.Spec can parse, for the specified kind
  # NB: ivars are unsupported
  def parse_objc_spec(decl:Cursor, nm:str, kind:CK) -> list[str]:
    nonlocal lines, types
    if decl is None: return []
    ms = []
    for d in filter(lambda d: d.kind == kind, decl.get_children()):
      rollback = lines, types
      try: ms.append(f"  ('{d.spelling}', {repr('instancetype') if (rt:=d.result_type).spelling=='instancetype' else tname(rt)}, "
        f"[{', '.join('instancetype' if a.spelling == 'instancetype' else tname(a.type) for a in d.get_arguments())}]" +
        (", True" if CK.NS_RETURNS_RETAINED in attrs(d) or (any(d.spelling.startswith(s) for s in arc_families) and rt.kind!=TK.VOID) else "") + "),")
      except NotImplementedError as e:
        print(f"skipping {nm}.{d.spelling}: {e}")
        lines, types = rollback
    return ms

  # libclang doesn't have a "type" for @protocol, so we have to do this here...
  def proto(decl):
    nonlocal lines, types
    if (nm:=decl.spelling) in types and types[nm][1]: return types[nm][0]
    # check if this is a forward declaration
    is_defn = bool([f.kind for f in decl.get_children() if f.kind in (CK.OBJC_INSTANCE_METHOD_DECL, CK.OBJC_CLASS_METHOD_DECL)])
    if nm not in types: lines.append(f"class {nm}(objc.Spec): pass")
    types[nm] = nm, is_defn
    if is_defn:
      bs = [proto(b) for b in decl.get_children() if b.kind==CK.OBJC_PROTOCOL_REF and b.spelling != decl.spelling]
      ims, cms = parse_objc_spec(decl, nm, CK.OBJC_INSTANCE_METHOD_DECL), parse_objc_spec(decl, nm, CK.OBJC_CLASS_METHOD_DECL)
      lines.extend([*([f"{nm}._bases_ = [{', '.join(bs)}]"] if bs else []),
                    *([f"{nm}._methods_ = [", *ims, "]"] if ims else []), *([f"{nm}._classmethods_ = [", *cms, "]"] if cms else [])])
    return nm

  for f in files:
    tu = Index.create().parse(f, args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
    (pp:=PP.create(tu.cursor)).set_property(PPP.TerseOutput, 1)
    for c in tu.cursor.walk_preorder():
      if str(c.location.file) != str(f) and (not recsym or c.kind not in (CK.FUNCTION_DECL,)): continue
      rollback = lines, types
      try:
        match c.kind:
          case CK.FUNCTION_DECL if c.linkage == LK.EXTERNAL and dll:
            # TODO: we could support name-mangling
            lines.append(f"# {c.pretty_printed(pp)}\ntry: ({c.spelling}:=dll.{c.spelling}).restype, {c.spelling}.argtypes = "
              f"{tname(c.result_type)}, [{', '.join(tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
            if CK.NS_RETURNS_RETAINED in attrs(c): lines.append(f"{c.spelling} = objc.returns_retained({c.spelling})")
          case CK.STRUCT_DECL | CK.UNION_DECL | CK.TYPEDEF_DECL | CK.ENUM_DECL | CK.OBJC_INTERFACE_DECL: tname(c.type)
          case CK.MACRO_DEFINITION if parse_macros and len(toks:=list(c.get_tokens())) > 1:
            if toks[1].spelling == '(' and toks[0].extent.end.column == toks[1].extent.start.column:
              it = iter(toks[1:])
              _args = [t.spelling for t in itertools.takewhile(lambda t:t.spelling!=')', it) if t.kind == ToK.IDENTIFIER]
              if len(body:=list(it)) == 0: continue
              macros += [f"{c.spelling} = lambda {','.join(_args)}: {readext(f, body[0].location.offset, toks[-1].extent.end.offset)}"]
            else: macros += [f"{c.spelling} = {readext(f, toks[1].location.offset, toks[-1].extent.end.offset)}"]
          case CK.VAR_DECL if c.linkage == LK.INTERNAL:
            if (c.type.kind == TK.CONSTANTARRAY and c.type.get_array_element_type().get_canonical().kind in ints and
                (init:=last(c)).kind == CK.INIT_LIST_EXPR and all(re.match(r"\[.*\].*=", readext(f, c.extent)) for c in init.get_children())):
              cs = init.get_children()
              macros += [f"{c.spelling} = {{{','.join(f'{readext(f,next(it:=c.get_children()).extent)}:{readext(f,next(it).extent)}' for c in cs)}}}"]
            elif c.type.get_canonical().kind in ints: macros += [f"{c.spelling} = {readext(f, last(c).extent)}"]
            else: macros += [f"{c.spelling} = {tname(c.type)}({readext(f, last(c).extent)})"]
          case CK.VAR_DECL if c.linkage == LK.EXTERNAL and dll:
            lines.append(f"try: {c.spelling} = {tname(c.type)}.in_dll(dll, '{c.spelling}')\nexcept (ValueError,AttributeError): pass")
          case CK.OBJC_PROTOCOL_DECL: proto(c)
      except NotImplementedError as e:
        print(f"skipping {c.spelling}: {e}")
        lines, types = rollback
  main = (f"# mypy: ignore-errors\nimport ctypes{', os' if any('os' in s for s in dll) else ''}\n"
    "from tinygrad.helpers import unwrap\nfrom tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR\n" + '\n'.join([*prolog,
      *(["from ctypes.util import find_library"]*any('find_library' in s for s in dll)), *(["from tinygrad.runtime.support import objc"]*objc),
      *(["def dll():",*flatten([[f"  try: return ctypes.CDLL(unwrap({d}){', use_errno=True' if use_errno else ''})",'  except: pass'] for d in dll]),
         "  return None", "dll = dll()\n"]*bool(dll)), *lines]) + '\n')
  macros = [r for m in macros if (r:=functools.reduce(lambda s,r:re.sub(r[0], r[1], s), rules + base_rules, m))]
  while True:
    try:
      exec(main + '\n'.join(macros), {})
      break
    except (SyntaxError, NameError, TypeError) as e:
      macrono = unwrap(e.lineno if isinstance(e, SyntaxError) else unwrap(unwrap(e.__traceback__).tb_next).tb_lineno) - main.count('\n') - 1
      assert macrono >= 0 and macrono < len(macros), f"error outside macro range: {e}"
      print(f"skipping {macros[macrono]}: {e}")
      del macros[macrono]
    except Exception as e: raise Exception("parsing failed") from e
  return main + '\n'.join(macros + epilog)
