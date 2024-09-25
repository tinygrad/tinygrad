# heavily inspired by dmbfm/zig-metal-gen
# NOTES
# in objc an interface is the spec for an implementation
# a protocol is a interface (as in like java)
#
# hence we search for interfaces to create classes & for protocols to generate mixins

from ctypes import c_int, c_uint
from dataclasses import dataclass
from clang.cindex import Config, Cursor, TranslationUnit, CursorKind, Type, TypeKind, conf

clang_to_ctypes_type = {
    TypeKind.INT: "c_int",
    TypeKind.UINT: "c_uint",
    TypeKind.FLOAT: "c_float",
    TypeKind.DOUBLE: "c_double",
    TypeKind.ULONG: "c_ulong",
    TypeKind.LONG: "c_long",
    TypeKind.ULONGLONG: "c_ulonglong",
    TypeKind.LONGLONG: "c_longlong",
    TypeKind.CHAR_S: "c_byte"
}

TypeKind.OBJCOBJECT = TypeKind(161)  # not declared in the python bindings for some reason

@dataclass
class ClassToGen:
    name: str
    superclass: str | None
    conforms: list[str]

classes_to_gen: dict[str, ClassToGen] = {}

def get_type(t: Type) -> str | None:
    kind: TypeKind = t.kind
    cannonical_kind = t.get_canonical().kind
    if cannonical_kind in clang_to_ctypes_type:
        return clang_to_ctypes_type[cannonical_kind]
    elif kind == TypeKind.OBJCOBJECTPOINTER:
        return get_type(t.get_pointee())  # in python, ClassName is the type of a pointer to this object
    elif kind == TypeKind.POINTER:
        pointee = t.get_pointee()
        if pointee.kind == TypeKind.CHAR_S: return "c_char_p"
        return f"ctypes.POINTER({get_type(pointee)})"
    elif kind == TypeKind.OBJCOBJECT:
        base: Type = conf.lib.clang_Type_getObjCObjectBaseType(t)
        num_protocols = conf.lib.clang_Type_getNumObjCProtocolRefs(t)
        if num_protocols == 1:
            protocol_cursor: Cursor = conf.lib.clang_Type_getObjCProtocolDecl(t, 0)
            print(protocol_cursor.kind)
            name = protocol_cursor.displayname
            if name not in classes_to_gen:
                for c in protocol_cursor.get_children():
                    c: Cursor = c
                    print(c.kind)
                classes_to_gen[name] = ClassToGen(name, None, [])
                print(classes_to_gen[name])
            return name
        else:
            return base.spelling
    elif kind == TypeKind.VOID:
        return None
    elif kind == TypeKind.OBJCINTERFACE:
        # add the interface to interfaces to gen if it's used
        print("TypeKind.OBJCINTERFACE", t.spelling)
        if t.spelling not in classes_to_gen:
            decl: Cursor = t.get_declaration()
            superclass: str | None = next(iter([c.spelling for c in decl.get_children() if c.kind == CursorKind.OBJC_SUPER_CLASS_REF]), None)
            conforms_to: list[str] = [c.spelling for c in decl.get_children() if c.kind == CursorKind.OBJC_PROTOCOL_REF]
            classes_to_gen[t.spelling] = ClassToGen(t.spelling, superclass, conforms_to)
            # print(classes_to_gen[t.spelling])
        return t.spelling
    elif kind == TypeKind.ELABORATED:
        return t.spelling

    raise Exception(f"couldn't resolve to a type t={t.spelling} {t.kind=}")

"""
# for alternative constructors
class Object:
    def __init__(self, name):
        self._ptr = ptr_from_name(name)
    @classmethod
    def from_ptr(cls, ptr):
        obj = cls.__new__(cls)
        super(MyClass, obj).__init__()
        obj._ptr = ptr
        return obj
"""

"""
class ObjcObject:
    def __init__(self, ptr):  # ctypes -> python
        assert ptr != None, "can't create class instance from null ptr"
        self._ptr = ptr
    def from_param(self):  # python -> ctypes
        return self_ptr
"""

def handle_function(cursor: Cursor):
    name = cursor.displayname.split("(")[0]  # name is "funName(type, type2)" => split to get the name
    if not name.startswith("MTL"):
        return
    ret = get_type(cursor.result_type)
    args: tuple[str, str] = [(t.spelling, get_type(t.type)) for t in cursor.get_arguments()]
    ret_annotation = f" -> {ret}" if ret is not None else ""
    print(f"metal.{name}.argtypes = [{', '.join(t for _, t in args)}]")
    print(f"metal.{name}.restype = {ret}")
    print(f"def {name}({', '.join([f'{n}: {t}' for n, t in args])}){ret_annotation}:")
    print(f"\treturn metal.{name}({', '.join([n for n, _ in args])})")

def handle_struct(cursor: Cursor):
    name = cursor.displayname
    if not name.startswith("MTL"):
        return
    childs = [c for c in cursor.get_children() if c.kind == CursorKind.FIELD_DECL]
    fields_str = ", ".join([f"(\"{c.spelling}\", {get_type(c.type)})" for c in childs])
    print(f"class {name}(Structure):")
    print(f"\t_fields_ = {fields_str}")

enums = {}

def handle_enum(cursor: Cursor):
    name = cursor.displayname
    if not name.startswith("MTL"):
        return
    if len(list(cursor.get_children())) == 0:
        return
    vs = set((c.spelling, c.enum_value) for c in cursor.get_children() if c.kind == CursorKind.ENUM_CONSTANT_DECL)
    if name not in enums:
        enums[name] = vs
    else:
        enums[name].update(vs)
    # type = get_type(cursor.enum_type) # in case we ever need it

def handle_protocol(cursor: Cursor):
    name = cursor.displayname
    if not (name.startswith("MTL") or name.startswith("NS")):
        return
    # print(f"class {name}Mixin:")
    for child in cursor.get_children():
        child: Cursor = child
        cdname = child.displayname
        ckind = child.kind
        # if ckind == CursorKind.OBJC_PROPERTY_DECL:
        #     restype = get_type(child.type)
        #     print("\t@property")
        #     print(f"\tdef {cdname}(self) -> {restype}:")
        #     print(f'\t\treturn send_msg(self, "{cdname}", restype={restype})')
        if ckind == CursorKind.OBJC_INSTANCE_METHOD_DECL:
            restype = get_type(child.result_type)
            argtypes = [get_type(t.type) for t in child.get_arguments()]
            argtypes_str = f'[{", ".join(str(a) for a in argtypes)}]'
            ret_annotation = f" -> {restype}" if restype is not None else ""
            return_instruction = "return " if restype is not None else ""
            # print(f"\tdef {cdname.replace(':', '_')}(){ret_annotation}:")
            # print(f'\t\t{return_instruction}send_msg(self, "{cdname}", restype={restype}, argtypes={argtypes_str})')
        # elif ckind == CursorKind.OBJC_CLASS_METHOD_DECL:
        #     print(f"\t{cdname} {ckind}")

def handle_record(cursor: Cursor):
    name = cursor.displayname
    print(name)
    if not (name.startswith("MTL") or name.startswith("NS")):
        return
    print(name)

def handle_interface(cursor: Cursor):
    name = cursor.displayname
    kind = cursor.kind
    # if kind == CursorKind.OBJC_CATEGORY_DECL:
    #     print(name)
    if not (name.startswith("MTL") or name.startswith("NS")):
        return
    # mixins = [get_type(c.type) for c in cursor.get_children() if c.kind in [CursorKind.OBJC_SUPER_CLASS_REF, CursorKind.OBJC_PROTOCOL_REF]]
    mixins = []
    # print(f'class {name}({"".join(f"{m}Mixin" for m in mixins)}):')
    # for c in cursor.get_children():
    #     tk = c.type.kind
    #     if c.kind == CursorKind.OBJC_SUPER_CLASS_REF:
    #         print(f"super: {c.displayname} {tk}")
    #     elif c.kind == CursorKind.OBJC_PROTOCOL_REF:
    #         print(f"implements: {c.displayname} {tk}")
    # print("\tpass")


# IF PROTOCOL IS USED AS ARG / RET, GENERATE CLASS(OBJ, MIXINS)

def traverse(node: Cursor, depth=0):
    try:
        k = node.kind
        if k == CursorKind.FUNCTION_DECL:
            # handle_function(node)
            pass
        elif k == CursorKind.STRUCT_DECL:
            # handle_struct(node)
            pass
        elif k == CursorKind.ENUM_DECL:
            # handle_enum(node)
            pass
        elif k == CursorKind.OBJC_PROTOCOL_DECL:
            # print(k, node.kind, node.displayname)
            handle_protocol(node)
            pass
        elif k == CursorKind.OBJC_CATEGORY_DECL or k == CursorKind.OBJC_INTERFACE_DECL:
            handle_interface(node)
            pass

    except ValueError as e:
        if "Unknown template argument kind" not in str(e):
            print(e.__class__, e)
    except Exception as e:
        print(e.__class__, e)
    for child in node.get_children(): traverse(child, depth + 1)

if __name__ == "__main__":
    Config.set_library_file("/opt/homebrew/Cellar/llvm/18.1.5/lib/libclang.dylib")
    conf.lib.clang_Type_getObjCObjectBaseType.argtypes = [Type]
    conf.lib.clang_Type_getObjCObjectBaseType.restype = Type
    conf.lib.clang_Type_getNumObjCProtocolRefs.argtypes = [Type]
    conf.lib.clang_Type_getNumObjCProtocolRefs.restype = c_int
    conf.lib.clang_Type_getObjCProtocolDecl.argtypes = [Type, c_uint]
    conf.lib.clang_Type_getObjCProtocolDecl.restype = Cursor

    tu = TranslationUnit.from_source("./entry.m")
    traverse(tu.cursor)

    # enums because enum decls can be duplicates for some reason
    for name, enum in enums.items():
        print(f"Class {name}:")
        for k, v in enum:
            print(f"\t{k} = {v}")
