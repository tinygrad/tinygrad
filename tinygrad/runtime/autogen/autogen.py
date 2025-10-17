import argparse
from clang.cindex import Config, Index, CursorKind, TypeKind

class Autogen:
  def __init__(self, lib, files, inc_dirs=[], clang_args=[]):
    self.index, self.lib, self.files, self.args, self.resolved = Index.create(), lib, files, [f"-I{i}" for i in inc_dirs] + clang_args, set()
  def run(self):
    print(f"import ctypes\ndll = ctypes.CDLL('{self.lib}')\n")
    for f in self.files:
      for c in self.index.parse(f, self.args).cursor.walk_preorder():
        try:
          match c.kind:
            case CursorKind.FUNCTION_DECL:
              print(f"# {c.displayname}\ntry: ({c.spelling}:=dll.{c.spelling}).restype, {c.spelling}.argtypes = {self.tname(c.result_type)}, "\
                    f"[{', '.join(self.tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
        except Exception as e: raise Exception(f"{e} at {c.location}") from e

  tmap = {TypeKind.VOID:"None", TypeKind.CHAR_U:"ctypes.c_ubyte", TypeKind.UCHAR:"ctypes.c_ubyte", TypeKind.CHAR_S:"ctypes.c_char",
          TypeKind.SCHAR:"ctypes.c_char", **{getattr(TypeKind, k):f"ctypes.c_{k.lower()}"
          for k in ["BOOL", "USHORT", "UINT", "ULONG", "ULONGLONG", "WCHAR", "SHORT", "INT", "LONG", "LONGLONG", "FLOAT", "DOUBLE", "LONGDOUBLE"]}}
  def tname(self, t) -> str:
    if t.kind in self.tmap: return self.tmap[t.kind]
    match t.kind:
      case TypeKind.POINTER: return f"ctypes.POINTER({self.tname(t.get_pointee())})"
      case TypeKind.ELABORATED: return self.tname(t.get_named_type())
      case TypeKind.TYPEDEF:
        if t.spelling not in self.resolved:
          self.resolved.add(t.spelling)
          print(f"{t.spelling} = {self.tname(t.get_canonical())}")
        return t.spelling
      case TypeKind.RECORD:
        if t.spelling not in self.resolved:
          self.resolved.add(t.spelling)
          decl = t.get_declaration()
          print(f"class {t.spelling.replace(' ', '_')}(ctypes.{'Structure' if decl.kind == CursorKind.STRUCT_DECL else 'Union'}):\n"\
                f"  _fields_ = [" + (", " if len(fs:=list(t.get_fields())) < 3 else ",\n              ").join(f"('{f.spelling}', {self.tname(f.type)})" for f in fs) + "]\n")
        return t.spelling.replace(" ", "_")
      case TypeKind.FUNCTIONPROTO: return f"ctypes.CFUNCTYPE({self.tname(t.get_result())}" + ((', ' + ', '.join(self.tname(arg) for arg in args))
                                                                                                 if (args:=t.argument_types()) else '') + ")"
      case TypeKind.CONSTANTARRAY: return f"({self.tname(t.get_array_element_type())} * {t.get_array_size()})"
      case _: raise NotImplementedError(f"{t.kind}")

if __name__ == "__main__":
  Config.set_library_file("/opt/homebrew/opt/llvm@20/lib/libclang.dylib")
  parser = argparse.ArgumentParser()
  parser.add_argument("FILE")
  parser.add_argument("-l", dest="lib")
  parser.add_argument("-I", dest="inc_dirs", action="append", default=[])
  parser.add_argument("--clang-args", nargs='*')

  args = parser.parse_args()

  Autogen(args.lib, [args.FILE], args.inc_dirs, args.clang_args or []).run()

