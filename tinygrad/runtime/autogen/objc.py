# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes, ctypes.util, os


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 8:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*8

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libobjc'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libobjc'] = ctypes.CDLL(ctypes.util.find_library('objc')) #  ctypes.CDLL('libobjc')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





_OBJC_OBJC_H_ = True # macro
OBJC_BOOL_IS_BOOL = 1 # macro
OBJC_BOOL_DEFINED = True # macro
YES = True # macro
NO = True # macro
# Nil = __DARWIN_NULL # macro
# nil = __DARWIN_NULL # macro
__autoreleasing = True # macro
class struct_objc_class(Structure):
    pass

Class = ctypes.c_void_p
class struct_objc_object(Structure):
    pass

struct_objc_object._pack_ = 1 # source:False
struct_objc_object._fields_ = [
    ('isa', ctypes.POINTER(struct_objc_class)),
]

id = ctypes.c_void_p
class struct_objc_selector(Structure):
    pass

SEL = ctypes.POINTER(struct_objc_selector)
IMP = ctypes.CFUNCTYPE(None)
BOOL = ctypes.c_bool
class struct__malloc_zone_t(Structure):
    pass

objc_zone_t = ctypes.POINTER(struct__malloc_zone_t)
try:
    sel_getName = _libraries['libobjc'].sel_getName
    sel_getName.restype = ctypes.POINTER(ctypes.c_char)
    sel_getName.argtypes = [SEL]
except AttributeError:
    pass
try:
    sel_registerName = _libraries['libobjc'].sel_registerName
    sel_registerName.restype = SEL
    sel_registerName.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    object_getClassName = _libraries['libobjc'].object_getClassName
    object_getClassName.restype = ctypes.POINTER(ctypes.c_char)
    object_getClassName.argtypes = [id]
except AttributeError:
    pass
try:
    object_getIndexedIvars = _libraries['libobjc'].object_getIndexedIvars
    object_getIndexedIvars.restype = ctypes.POINTER(None)
    object_getIndexedIvars.argtypes = [id]
except AttributeError:
    pass
try:
    sel_isMapped = _libraries['libobjc'].sel_isMapped
    sel_isMapped.restype = BOOL
    sel_isMapped.argtypes = [SEL]
except AttributeError:
    pass
try:
    sel_getUid = _libraries['libobjc'].sel_getUid
    sel_getUid.restype = SEL
    sel_getUid.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
objc_objectptr_t = ctypes.POINTER(None)
try:
    objc_retainedObject = _libraries['libobjc'].objc_retainedObject
    objc_retainedObject.restype = id
    objc_retainedObject.argtypes = [objc_objectptr_t]
except AttributeError:
    pass
try:
    objc_unretainedObject = _libraries['libobjc'].objc_unretainedObject
    objc_unretainedObject.restype = id
    objc_unretainedObject.argtypes = [objc_objectptr_t]
except AttributeError:
    pass
try:
    objc_unretainedPointer = _libraries['libobjc'].objc_unretainedPointer
    objc_unretainedPointer.restype = objc_objectptr_t
    objc_unretainedPointer.argtypes = [id]
except AttributeError:
    pass
_OBJC_MESSAGE_H = True # macro
OBJC_SUPER = True # macro
class struct_objc_super(Structure):
    pass

struct_objc_super._pack_ = 1 # source:False
struct_objc_super._fields_ = [
    ('receiver', ctypes.POINTER(struct_objc_object)),
    ('super_class', ctypes.POINTER(struct_objc_class)),
]

try:
    objc_msgSend = _libraries['libobjc'].objc_msgSend
    objc_msgSend.restype = None
    objc_msgSend.argtypes = []
except AttributeError:
    pass
try:
    objc_msgSendSuper = _libraries['libobjc'].objc_msgSendSuper
    objc_msgSendSuper.restype = None
    objc_msgSendSuper.argtypes = []
except AttributeError:
    pass
try:
    objc_msgSend_stret = _libraries['libobjc'].objc_msgSend_stret
    objc_msgSend_stret.restype = None
    objc_msgSend_stret.argtypes = []
except AttributeError:
    pass
try:
    objc_msgSendSuper_stret = _libraries['libobjc'].objc_msgSendSuper_stret
    objc_msgSendSuper_stret.restype = None
    objc_msgSendSuper_stret.argtypes = []
except AttributeError:
    pass
try:
    method_invoke = _libraries['libobjc'].method_invoke
    method_invoke.restype = None
    method_invoke.argtypes = []
except AttributeError:
    pass
try:
    method_invoke_stret = _libraries['libobjc'].method_invoke_stret
    method_invoke_stret.restype = None
    method_invoke_stret.argtypes = []
except AttributeError:
    pass
try:
    _objc_msgForward = _libraries['libobjc']._objc_msgForward
    _objc_msgForward.restype = None
    _objc_msgForward.argtypes = []
except AttributeError:
    pass
try:
    _objc_msgForward_stret = _libraries['libobjc']._objc_msgForward_stret
    _objc_msgForward_stret.restype = None
    _objc_msgForward_stret.argtypes = []
except AttributeError:
    pass
_OBJC_RUNTIME_H = True # macro
# OBJC_DYNAMIC_CLASSES = ((void*)-1) # macro
OBJC_GETCLASSHOOK_DEFINED = 1 # macro
OBJC_ADDLOADIMAGEFUNC_DEFINED = 1 # macro
OBJC_SETHOOK_LAZYCLASSNAMER_DEFINED = 1 # macro
OBJC_REALIZECLASSFROMSWIFT_DEFINED = 1 # macro
_C_ID = '@' # macro
_C_CLASS = '#' # macro
# _C_SEL = ':' # macro
_C_CHR = 'c' # macro
_C_UCHR = 'C' # macro
_C_SHT = 's' # macro
_C_USHT = 'S' # macro
_C_INT = 'i' # macro
_C_UINT = 'I' # macro
_C_LNG = 'l' # macro
_C_ULNG = 'L' # macro
_C_LNG_LNG = 'q' # macro
_C_ULNG_LNG = 'Q' # macro
_C_INT128 = 't' # macro
_C_UINT128 = 'T' # macro
_C_FLT = 'f' # macro
_C_DBL = 'd' # macro
_C_LNG_DBL = 'D' # macro
_C_BFLD = 'b' # macro
_C_BOOL = 'B' # macro
_C_VOID = 'v' # macro
# _C_UNDEF = '?' # macro
_C_PTR = '^' # macro
_C_CHARPTR = '*' # macro
_C_ATOM = '%' # macro
_C_ARY_B = '[' # macro
_C_ARY_E = ']' # macro
# _C_UNION_B = '(' # macro
# _C_UNION_E = ')' # macro
_C_STRUCT_B = '{' # macro
_C_STRUCT_E = '}' # macro
_C_VECTOR = '!' # macro
_C_COMPLEX = 'j' # macro
_C_ATOMIC = 'A' # macro
_C_CONST = 'r' # macro
_C_IN = 'n' # macro
_C_INOUT = 'N' # macro
_C_OUT = 'o' # macro
_C_BYCOPY = 'O' # macro
_C_BYREF = 'R' # macro
_C_ONEWAY = 'V' # macro
_C_GNUREGISTER = '+' # macro
class struct_objc_method(Structure):
    pass

Method = ctypes.POINTER(struct_objc_method)
class struct_objc_ivar(Structure):
    pass

Ivar = ctypes.POINTER(struct_objc_ivar)
class struct_objc_category(Structure):
    pass

Category = ctypes.POINTER(struct_objc_category)
class struct_objc_property(Structure):
    pass

objc_property_t = ctypes.POINTER(struct_objc_property)
Protocol = struct_objc_object
class struct_objc_method_description(Structure):
    pass

struct_objc_method_description._pack_ = 1 # source:False
struct_objc_method_description._fields_ = [
    ('name', ctypes.POINTER(struct_objc_selector)),
    ('types', ctypes.POINTER(ctypes.c_char)),
]

class struct_objc_property_attribute_t(Structure):
    pass

struct_objc_property_attribute_t._pack_ = 1 # source:False
struct_objc_property_attribute_t._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('value', ctypes.POINTER(ctypes.c_char)),
]

objc_property_attribute_t = struct_objc_property_attribute_t
class struct_mach_header(Structure):
    pass

size_t = ctypes.c_uint64
try:
    object_copy = _libraries['libobjc'].object_copy
    object_copy.restype = id
    object_copy.argtypes = [id, size_t]
except AttributeError:
    pass
try:
    object_dispose = _libraries['libobjc'].object_dispose
    object_dispose.restype = id
    object_dispose.argtypes = [id]
except AttributeError:
    pass
try:
    object_getClass = _libraries['libobjc'].object_getClass
    object_getClass.restype = Class
    object_getClass.argtypes = [id]
except AttributeError:
    pass
try:
    object_setClass = _libraries['libobjc'].object_setClass
    object_setClass.restype = Class
    object_setClass.argtypes = [id, Class]
except AttributeError:
    pass
try:
    object_isClass = _libraries['libobjc'].object_isClass
    object_isClass.restype = BOOL
    object_isClass.argtypes = [id]
except AttributeError:
    pass
try:
    object_getIvar = _libraries['libobjc'].object_getIvar
    object_getIvar.restype = id
    object_getIvar.argtypes = [id, Ivar]
except AttributeError:
    pass
try:
    object_setIvar = _libraries['libobjc'].object_setIvar
    object_setIvar.restype = None
    object_setIvar.argtypes = [id, Ivar, id]
except AttributeError:
    pass
try:
    object_setIvarWithStrongDefault = _libraries['libobjc'].object_setIvarWithStrongDefault
    object_setIvarWithStrongDefault.restype = None
    object_setIvarWithStrongDefault.argtypes = [id, Ivar, id]
except AttributeError:
    pass
try:
    object_setInstanceVariable = _libraries['libobjc'].object_setInstanceVariable
    object_setInstanceVariable.restype = Ivar
    object_setInstanceVariable.argtypes = [id, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    object_setInstanceVariableWithStrongDefault = _libraries['libobjc'].object_setInstanceVariableWithStrongDefault
    object_setInstanceVariableWithStrongDefault.restype = Ivar
    object_setInstanceVariableWithStrongDefault.argtypes = [id, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    object_getInstanceVariable = _libraries['libobjc'].object_getInstanceVariable
    object_getInstanceVariable.restype = Ivar
    object_getInstanceVariable.argtypes = [id, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    objc_getClass = _libraries['libobjc'].objc_getClass
    objc_getClass.restype = Class
    objc_getClass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_getMetaClass = _libraries['libobjc'].objc_getMetaClass
    objc_getMetaClass.restype = Class
    objc_getMetaClass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_lookUpClass = _libraries['libobjc'].objc_lookUpClass
    objc_lookUpClass.restype = Class
    objc_lookUpClass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_getRequiredClass = _libraries['libobjc'].objc_getRequiredClass
    objc_getRequiredClass.restype = Class
    objc_getRequiredClass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_getClassList = _libraries['libobjc'].objc_getClassList
    objc_getClassList.restype = ctypes.c_int32
    objc_getClassList.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_objc_class)), ctypes.c_int32]
except AttributeError:
    pass
try:
    objc_copyClassList = _libraries['libobjc'].objc_copyClassList
    objc_copyClassList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_class))
    objc_copyClassList.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    objc_enumerateClasses = _libraries['libobjc'].objc_enumerateClasses
    objc_enumerateClasses.restype = None
    objc_enumerateClasses.argtypes = [ctypes.POINTER(None), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_object), Class, ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_objc_class), ctypes.POINTER(ctypes.c_bool))]
except AttributeError:
    pass
try:
    class_getName = _libraries['libobjc'].class_getName
    class_getName.restype = ctypes.POINTER(ctypes.c_char)
    class_getName.argtypes = [Class]
except AttributeError:
    pass
try:
    class_isMetaClass = _libraries['libobjc'].class_isMetaClass
    class_isMetaClass.restype = BOOL
    class_isMetaClass.argtypes = [Class]
except AttributeError:
    pass
try:
    class_getSuperclass = _libraries['libobjc'].class_getSuperclass
    class_getSuperclass.restype = Class
    class_getSuperclass.argtypes = [Class]
except AttributeError:
    pass
try:
    class_setSuperclass = _libraries['libobjc'].class_setSuperclass
    class_setSuperclass.restype = Class
    class_setSuperclass.argtypes = [Class, Class]
except AttributeError:
    pass
try:
    class_getVersion = _libraries['libobjc'].class_getVersion
    class_getVersion.restype = ctypes.c_int32
    class_getVersion.argtypes = [Class]
except AttributeError:
    pass
try:
    class_setVersion = _libraries['libobjc'].class_setVersion
    class_setVersion.restype = None
    class_setVersion.argtypes = [Class, ctypes.c_int32]
except AttributeError:
    pass
try:
    class_getInstanceSize = _libraries['libobjc'].class_getInstanceSize
    class_getInstanceSize.restype = size_t
    class_getInstanceSize.argtypes = [Class]
except AttributeError:
    pass
try:
    class_getInstanceVariable = _libraries['libobjc'].class_getInstanceVariable
    class_getInstanceVariable.restype = Ivar
    class_getInstanceVariable.argtypes = [Class, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_getClassVariable = _libraries['libobjc'].class_getClassVariable
    class_getClassVariable.restype = Ivar
    class_getClassVariable.argtypes = [Class, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_copyIvarList = _libraries['libobjc'].class_copyIvarList
    class_copyIvarList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_ivar))
    class_copyIvarList.argtypes = [Class, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    class_getInstanceMethod = _libraries['libobjc'].class_getInstanceMethod
    class_getInstanceMethod.restype = Method
    class_getInstanceMethod.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_getClassMethod = _libraries['libobjc'].class_getClassMethod
    class_getClassMethod.restype = Method
    class_getClassMethod.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_getMethodImplementation = _libraries['libobjc'].class_getMethodImplementation
    class_getMethodImplementation.restype = IMP
    class_getMethodImplementation.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_getMethodImplementation_stret = _libraries['libobjc'].class_getMethodImplementation_stret
    class_getMethodImplementation_stret.restype = IMP
    class_getMethodImplementation_stret.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_respondsToSelector = _libraries['libobjc'].class_respondsToSelector
    class_respondsToSelector.restype = BOOL
    class_respondsToSelector.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_copyMethodList = _libraries['libobjc'].class_copyMethodList
    class_copyMethodList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_method))
    class_copyMethodList.argtypes = [Class, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    class_conformsToProtocol = _libraries['libobjc'].class_conformsToProtocol
    class_conformsToProtocol.restype = BOOL
    class_conformsToProtocol.argtypes = [Class, ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    class_copyProtocolList = _libraries['libobjc'].class_copyProtocolList
    class_copyProtocolList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_object))
    class_copyProtocolList.argtypes = [Class, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    class_getProperty = _libraries['libobjc'].class_getProperty
    class_getProperty.restype = objc_property_t
    class_getProperty.argtypes = [Class, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_copyPropertyList = _libraries['libobjc'].class_copyPropertyList
    class_copyPropertyList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_property))
    class_copyPropertyList.argtypes = [Class, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    class_getIvarLayout = _libraries['libobjc'].class_getIvarLayout
    class_getIvarLayout.restype = ctypes.POINTER(ctypes.c_ubyte)
    class_getIvarLayout.argtypes = [Class]
except AttributeError:
    pass
try:
    class_getWeakIvarLayout = _libraries['libobjc'].class_getWeakIvarLayout
    class_getWeakIvarLayout.restype = ctypes.POINTER(ctypes.c_ubyte)
    class_getWeakIvarLayout.argtypes = [Class]
except AttributeError:
    pass
try:
    class_addMethod = _libraries['libobjc'].class_addMethod
    class_addMethod.restype = BOOL
    class_addMethod.argtypes = [Class, SEL, IMP, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_replaceMethod = _libraries['libobjc'].class_replaceMethod
    class_replaceMethod.restype = IMP
    class_replaceMethod.argtypes = [Class, SEL, IMP, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
uint8_t = ctypes.c_uint8
try:
    class_addIvar = _libraries['libobjc'].class_addIvar
    class_addIvar.restype = BOOL
    class_addIvar.argtypes = [Class, ctypes.POINTER(ctypes.c_char), size_t, uint8_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_addProtocol = _libraries['libobjc'].class_addProtocol
    class_addProtocol.restype = BOOL
    class_addProtocol.argtypes = [Class, ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    class_addProperty = _libraries['libobjc'].class_addProperty
    class_addProperty.restype = BOOL
    class_addProperty.argtypes = [Class, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_property_attribute_t), ctypes.c_uint32]
except AttributeError:
    pass
try:
    class_replaceProperty = _libraries['libobjc'].class_replaceProperty
    class_replaceProperty.restype = None
    class_replaceProperty.argtypes = [Class, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_property_attribute_t), ctypes.c_uint32]
except AttributeError:
    pass
try:
    class_setIvarLayout = _libraries['libobjc'].class_setIvarLayout
    class_setIvarLayout.restype = None
    class_setIvarLayout.argtypes = [Class, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    class_setWeakIvarLayout = _libraries['libobjc'].class_setWeakIvarLayout
    class_setWeakIvarLayout.restype = None
    class_setWeakIvarLayout.argtypes = [Class, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    objc_getFutureClass = _libraries['libobjc'].objc_getFutureClass
    objc_getFutureClass.restype = Class
    objc_getFutureClass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    class_createInstance = _libraries['libobjc'].class_createInstance
    class_createInstance.restype = id
    class_createInstance.argtypes = [Class, size_t]
except AttributeError:
    pass
try:
    objc_constructInstance = _libraries['libobjc'].objc_constructInstance
    objc_constructInstance.restype = id
    objc_constructInstance.argtypes = [Class, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    objc_destructInstance = _libraries['libobjc'].objc_destructInstance
    objc_destructInstance.restype = ctypes.POINTER(None)
    objc_destructInstance.argtypes = [id]
except AttributeError:
    pass
try:
    objc_allocateClassPair = _libraries['libobjc'].objc_allocateClassPair
    objc_allocateClassPair.restype = Class
    objc_allocateClassPair.argtypes = [Class, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    objc_registerClassPair = _libraries['libobjc'].objc_registerClassPair
    objc_registerClassPair.restype = None
    objc_registerClassPair.argtypes = [Class]
except AttributeError:
    pass
try:
    objc_duplicateClass = _libraries['libobjc'].objc_duplicateClass
    objc_duplicateClass.restype = Class
    objc_duplicateClass.argtypes = [Class, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    objc_disposeClassPair = _libraries['libobjc'].objc_disposeClassPair
    objc_disposeClassPair.restype = None
    objc_disposeClassPair.argtypes = [Class]
except AttributeError:
    pass
try:
    method_getName = _libraries['libobjc'].method_getName
    method_getName.restype = SEL
    method_getName.argtypes = [Method]
except AttributeError:
    pass
try:
    method_getImplementation = _libraries['libobjc'].method_getImplementation
    method_getImplementation.restype = IMP
    method_getImplementation.argtypes = [Method]
except AttributeError:
    pass
try:
    method_getTypeEncoding = _libraries['libobjc'].method_getTypeEncoding
    method_getTypeEncoding.restype = ctypes.POINTER(ctypes.c_char)
    method_getTypeEncoding.argtypes = [Method]
except AttributeError:
    pass
try:
    method_getNumberOfArguments = _libraries['libobjc'].method_getNumberOfArguments
    method_getNumberOfArguments.restype = ctypes.c_uint32
    method_getNumberOfArguments.argtypes = [Method]
except AttributeError:
    pass
try:
    method_copyReturnType = _libraries['libobjc'].method_copyReturnType
    method_copyReturnType.restype = ctypes.POINTER(ctypes.c_char)
    method_copyReturnType.argtypes = [Method]
except AttributeError:
    pass
try:
    method_copyArgumentType = _libraries['libobjc'].method_copyArgumentType
    method_copyArgumentType.restype = ctypes.POINTER(ctypes.c_char)
    method_copyArgumentType.argtypes = [Method, ctypes.c_uint32]
except AttributeError:
    pass
try:
    method_getReturnType = _libraries['libobjc'].method_getReturnType
    method_getReturnType.restype = None
    method_getReturnType.argtypes = [Method, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    method_getArgumentType = _libraries['libobjc'].method_getArgumentType
    method_getArgumentType.restype = None
    method_getArgumentType.argtypes = [Method, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    method_getDescription = _libraries['libobjc'].method_getDescription
    method_getDescription.restype = ctypes.POINTER(struct_objc_method_description)
    method_getDescription.argtypes = [Method]
except AttributeError:
    pass
try:
    method_setImplementation = _libraries['libobjc'].method_setImplementation
    method_setImplementation.restype = IMP
    method_setImplementation.argtypes = [Method, IMP]
except AttributeError:
    pass
try:
    method_exchangeImplementations = _libraries['libobjc'].method_exchangeImplementations
    method_exchangeImplementations.restype = None
    method_exchangeImplementations.argtypes = [Method, Method]
except AttributeError:
    pass
try:
    ivar_getName = _libraries['libobjc'].ivar_getName
    ivar_getName.restype = ctypes.POINTER(ctypes.c_char)
    ivar_getName.argtypes = [Ivar]
except AttributeError:
    pass
try:
    ivar_getTypeEncoding = _libraries['libobjc'].ivar_getTypeEncoding
    ivar_getTypeEncoding.restype = ctypes.POINTER(ctypes.c_char)
    ivar_getTypeEncoding.argtypes = [Ivar]
except AttributeError:
    pass
ptrdiff_t = ctypes.c_int64
try:
    ivar_getOffset = _libraries['libobjc'].ivar_getOffset
    ivar_getOffset.restype = ptrdiff_t
    ivar_getOffset.argtypes = [Ivar]
except AttributeError:
    pass
try:
    property_getName = _libraries['libobjc'].property_getName
    property_getName.restype = ctypes.POINTER(ctypes.c_char)
    property_getName.argtypes = [objc_property_t]
except AttributeError:
    pass
try:
    property_getAttributes = _libraries['libobjc'].property_getAttributes
    property_getAttributes.restype = ctypes.POINTER(ctypes.c_char)
    property_getAttributes.argtypes = [objc_property_t]
except AttributeError:
    pass
try:
    property_copyAttributeList = _libraries['libobjc'].property_copyAttributeList
    property_copyAttributeList.restype = ctypes.POINTER(struct_objc_property_attribute_t)
    property_copyAttributeList.argtypes = [objc_property_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    property_copyAttributeValue = _libraries['libobjc'].property_copyAttributeValue
    property_copyAttributeValue.restype = ctypes.POINTER(ctypes.c_char)
    property_copyAttributeValue.argtypes = [objc_property_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_getProtocol = _libraries['libobjc'].objc_getProtocol
    objc_getProtocol.restype = ctypes.POINTER(struct_objc_object)
    objc_getProtocol.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_copyProtocolList = _libraries['libobjc'].objc_copyProtocolList
    objc_copyProtocolList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_object))
    objc_copyProtocolList.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    protocol_conformsToProtocol = _libraries['libobjc'].protocol_conformsToProtocol
    protocol_conformsToProtocol.restype = BOOL
    protocol_conformsToProtocol.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    protocol_isEqual = _libraries['libobjc'].protocol_isEqual
    protocol_isEqual.restype = BOOL
    protocol_isEqual.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    protocol_getName = _libraries['libobjc'].protocol_getName
    protocol_getName.restype = ctypes.POINTER(ctypes.c_char)
    protocol_getName.argtypes = [ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    protocol_getMethodDescription = _libraries['libobjc'].protocol_getMethodDescription
    protocol_getMethodDescription.restype = struct_objc_method_description
    protocol_getMethodDescription.argtypes = [ctypes.POINTER(struct_objc_object), SEL, BOOL, BOOL]
except AttributeError:
    pass
try:
    protocol_copyMethodDescriptionList = _libraries['libobjc'].protocol_copyMethodDescriptionList
    protocol_copyMethodDescriptionList.restype = ctypes.POINTER(struct_objc_method_description)
    protocol_copyMethodDescriptionList.argtypes = [ctypes.POINTER(struct_objc_object), BOOL, BOOL, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    protocol_getProperty = _libraries['libobjc'].protocol_getProperty
    protocol_getProperty.restype = objc_property_t
    protocol_getProperty.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(ctypes.c_char), BOOL, BOOL]
except AttributeError:
    pass
try:
    protocol_copyPropertyList = _libraries['libobjc'].protocol_copyPropertyList
    protocol_copyPropertyList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_property))
    protocol_copyPropertyList.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    protocol_copyPropertyList2 = _libraries['libobjc'].protocol_copyPropertyList2
    protocol_copyPropertyList2.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_property))
    protocol_copyPropertyList2.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(ctypes.c_uint32), BOOL, BOOL]
except AttributeError:
    pass
try:
    protocol_copyProtocolList = _libraries['libobjc'].protocol_copyProtocolList
    protocol_copyProtocolList.restype = ctypes.POINTER(ctypes.POINTER(struct_objc_object))
    protocol_copyProtocolList.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    objc_allocateProtocol = _libraries['libobjc'].objc_allocateProtocol
    objc_allocateProtocol.restype = ctypes.POINTER(struct_objc_object)
    objc_allocateProtocol.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    objc_registerProtocol = _libraries['libobjc'].objc_registerProtocol
    objc_registerProtocol.restype = None
    objc_registerProtocol.argtypes = [ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    protocol_addMethodDescription = _libraries['libobjc'].protocol_addMethodDescription
    protocol_addMethodDescription.restype = None
    protocol_addMethodDescription.argtypes = [ctypes.POINTER(struct_objc_object), SEL, ctypes.POINTER(ctypes.c_char), BOOL, BOOL]
except AttributeError:
    pass
try:
    protocol_addProtocol = _libraries['libobjc'].protocol_addProtocol
    protocol_addProtocol.restype = None
    protocol_addProtocol.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(struct_objc_object)]
except AttributeError:
    pass
try:
    protocol_addProperty = _libraries['libobjc'].protocol_addProperty
    protocol_addProperty.restype = None
    protocol_addProperty.argtypes = [ctypes.POINTER(struct_objc_object), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_property_attribute_t), ctypes.c_uint32, BOOL, BOOL]
except AttributeError:
    pass
try:
    objc_copyImageNames = _libraries['libobjc'].objc_copyImageNames
    objc_copyImageNames.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
    objc_copyImageNames.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    class_getImageName = _libraries['libobjc'].class_getImageName
    class_getImageName.restype = ctypes.POINTER(ctypes.c_char)
    class_getImageName.argtypes = [Class]
except AttributeError:
    pass
try:
    objc_copyClassNamesForImage = _libraries['libobjc'].objc_copyClassNamesForImage
    objc_copyClassNamesForImage.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
    objc_copyClassNamesForImage.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    sel_isEqual = _libraries['libobjc'].sel_isEqual
    sel_isEqual.restype = BOOL
    sel_isEqual.argtypes = [SEL, SEL]
except AttributeError:
    pass
try:
    objc_enumerationMutation = _libraries['libobjc'].objc_enumerationMutation
    objc_enumerationMutation.restype = None
    objc_enumerationMutation.argtypes = [id]
except AttributeError:
    pass
try:
    objc_setEnumerationMutationHandler = _libraries['libobjc'].objc_setEnumerationMutationHandler
    objc_setEnumerationMutationHandler.restype = None
    objc_setEnumerationMutationHandler.argtypes = [ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_objc_object))]
except AttributeError:
    pass
try:
    objc_setForwardHandler = _libraries['libobjc'].objc_setForwardHandler
    objc_setForwardHandler.restype = None
    objc_setForwardHandler.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    imp_implementationWithBlock = _libraries['libobjc'].imp_implementationWithBlock
    imp_implementationWithBlock.restype = IMP
    imp_implementationWithBlock.argtypes = [id]
except AttributeError:
    pass
try:
    imp_getBlock = _libraries['libobjc'].imp_getBlock
    imp_getBlock.restype = id
    imp_getBlock.argtypes = [IMP]
except AttributeError:
    pass
try:
    imp_removeBlock = _libraries['libobjc'].imp_removeBlock
    imp_removeBlock.restype = BOOL
    imp_removeBlock.argtypes = [IMP]
except AttributeError:
    pass
try:
    objc_loadWeak = _libraries['libobjc'].objc_loadWeak
    objc_loadWeak.restype = id
    objc_loadWeak.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_objc_object))]
except AttributeError:
    pass
try:
    objc_storeWeak = _libraries['libobjc'].objc_storeWeak
    objc_storeWeak.restype = id
    objc_storeWeak.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_objc_object)), id]
except AttributeError:
    pass
objc_AssociationPolicy = ctypes.c_uint64

# values for enumeration 'enum_runtime_h_1642'
enum_runtime_h_1642__enumvalues = {
    0: 'OBJC_ASSOCIATION_ASSIGN',
    1: 'OBJC_ASSOCIATION_RETAIN_NONATOMIC',
    3: 'OBJC_ASSOCIATION_COPY_NONATOMIC',
    769: 'OBJC_ASSOCIATION_RETAIN',
    771: 'OBJC_ASSOCIATION_COPY',
}
OBJC_ASSOCIATION_ASSIGN = 0
OBJC_ASSOCIATION_RETAIN_NONATOMIC = 1
OBJC_ASSOCIATION_COPY_NONATOMIC = 3
OBJC_ASSOCIATION_RETAIN = 769
OBJC_ASSOCIATION_COPY = 771
enum_runtime_h_1642 = ctypes.c_uint32 # enum
try:
    objc_setAssociatedObject = _libraries['libobjc'].objc_setAssociatedObject
    objc_setAssociatedObject.restype = None
    objc_setAssociatedObject.argtypes = [id, ctypes.POINTER(None), id, objc_AssociationPolicy]
except AttributeError:
    pass
try:
    objc_getAssociatedObject = _libraries['libobjc'].objc_getAssociatedObject
    objc_getAssociatedObject.restype = id
    objc_getAssociatedObject.argtypes = [id, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    objc_removeAssociatedObjects = _libraries['libobjc'].objc_removeAssociatedObjects
    objc_removeAssociatedObjects.restype = None
    objc_removeAssociatedObjects.argtypes = [id]
except AttributeError:
    pass
objc_hook_getImageName = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_objc_class), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
try:
    objc_setHook_getImageName = _libraries['libobjc'].objc_setHook_getImageName
    objc_setHook_getImageName.restype = None
    objc_setHook_getImageName.argtypes = [objc_hook_getImageName, ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_objc_class), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))]
except AttributeError:
    pass
objc_hook_getClass = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_objc_class)))
try:
    objc_setHook_getClass = _libraries['libobjc'].objc_setHook_getClass
    objc_setHook_getClass.restype = None
    objc_setHook_getClass.argtypes = [objc_hook_getClass, ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_objc_class))))]
except AttributeError:
    pass
objc_func_loadImage = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_mach_header))
try:
    objc_addLoadImageFunc = _libraries['libobjc'].objc_addLoadImageFunc
    objc_addLoadImageFunc.restype = None
    objc_addLoadImageFunc.argtypes = [objc_func_loadImage]
except AttributeError:
    pass
objc_hook_lazyClassNamer = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_class))
try:
    objc_setHook_lazyClassNamer = _libraries['libobjc'].objc_setHook_lazyClassNamer
    objc_setHook_lazyClassNamer.restype = None
    objc_setHook_lazyClassNamer.argtypes = [objc_hook_lazyClassNamer, ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_objc_class)))]
except AttributeError:
    pass
_objc_swiftMetadataInitializer = ctypes.CFUNCTYPE(ctypes.POINTER(struct_objc_class), ctypes.POINTER(struct_objc_class), ctypes.POINTER(None))
try:
    _objc_realizeClassFromSwift = _libraries['libobjc']._objc_realizeClassFromSwift
    _objc_realizeClassFromSwift.restype = Class
    _objc_realizeClassFromSwift.argtypes = [Class, ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_objc_method_list(Structure):
    pass

try:
    _objc_flush_caches = _libraries['libobjc']._objc_flush_caches
    _objc_flush_caches.restype = None
    _objc_flush_caches.argtypes = [Class]
except AttributeError:
    pass
try:
    class_lookupMethod = _libraries['libobjc'].class_lookupMethod
    class_lookupMethod.restype = IMP
    class_lookupMethod.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    class_respondsToMethod = _libraries['libobjc'].class_respondsToMethod
    class_respondsToMethod.restype = BOOL
    class_respondsToMethod.argtypes = [Class, SEL]
except AttributeError:
    pass
try:
    object_copyFromZone = _libraries['libobjc'].object_copyFromZone
    object_copyFromZone.restype = id
    object_copyFromZone.argtypes = [id, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    class_createInstanceFromZone = _libraries['libobjc'].class_createInstanceFromZone
    class_createInstanceFromZone.restype = id
    class_createInstanceFromZone.argtypes = [Class, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
__all__ = \
    ['BOOL', 'Category', 'Class', 'IMP', 'Ivar', 'Method', 'NO',
    'OBJC_ADDLOADIMAGEFUNC_DEFINED', 'OBJC_ASSOCIATION_ASSIGN',
    'OBJC_ASSOCIATION_COPY', 'OBJC_ASSOCIATION_COPY_NONATOMIC',
    'OBJC_ASSOCIATION_RETAIN', 'OBJC_ASSOCIATION_RETAIN_NONATOMIC',
    'OBJC_BOOL_DEFINED', 'OBJC_BOOL_IS_BOOL',
    'OBJC_GETCLASSHOOK_DEFINED', 'OBJC_REALIZECLASSFROMSWIFT_DEFINED',
    'OBJC_SETHOOK_LAZYCLASSNAMER_DEFINED', 'OBJC_SUPER', 'Protocol',
    'SEL', 'YES', '_C_ARY_B', '_C_ARY_E', '_C_ATOM', '_C_ATOMIC',
    '_C_BFLD', '_C_BOOL', '_C_BYCOPY', '_C_BYREF', '_C_CHARPTR',
    '_C_CHR', '_C_CLASS', '_C_COMPLEX', '_C_CONST', '_C_DBL',
    '_C_FLT', '_C_GNUREGISTER', '_C_ID', '_C_IN', '_C_INOUT',
    '_C_INT', '_C_INT128', '_C_LNG', '_C_LNG_DBL', '_C_LNG_LNG',
    '_C_ONEWAY', '_C_OUT', '_C_PTR', '_C_SHT', '_C_STRUCT_B',
    '_C_STRUCT_E', '_C_UCHR', '_C_UINT', '_C_UINT128', '_C_ULNG',
    '_C_ULNG_LNG', '_C_UNION_B', '_C_UNION_E', '_C_USHT', '_C_VECTOR',
    '_C_VOID', '_OBJC_MESSAGE_H', '_OBJC_OBJC_H_', '_OBJC_RUNTIME_H',
    '__autoreleasing', '_objc_flush_caches', '_objc_msgForward',
    '_objc_msgForward_stret', '_objc_realizeClassFromSwift',
    '_objc_swiftMetadataInitializer', 'class_addIvar',
    'class_addMethod', 'class_addProperty', 'class_addProtocol',
    'class_conformsToProtocol', 'class_copyIvarList',
    'class_copyMethodList', 'class_copyPropertyList',
    'class_copyProtocolList', 'class_createInstance',
    'class_createInstanceFromZone', 'class_getClassMethod',
    'class_getClassVariable', 'class_getImageName',
    'class_getInstanceMethod', 'class_getInstanceSize',
    'class_getInstanceVariable', 'class_getIvarLayout',
    'class_getMethodImplementation',
    'class_getMethodImplementation_stret', 'class_getName',
    'class_getProperty', 'class_getSuperclass', 'class_getVersion',
    'class_getWeakIvarLayout', 'class_isMetaClass',
    'class_lookupMethod', 'class_replaceMethod',
    'class_replaceProperty', 'class_respondsToMethod',
    'class_respondsToSelector', 'class_setIvarLayout',
    'class_setSuperclass', 'class_setVersion',
    'class_setWeakIvarLayout', 'enum_runtime_h_1642', 'id',
    'imp_getBlock', 'imp_implementationWithBlock', 'imp_removeBlock',
    'ivar_getName', 'ivar_getOffset', 'ivar_getTypeEncoding',
    'method_copyArgumentType', 'method_copyReturnType',
    'method_exchangeImplementations', 'method_getArgumentType',
    'method_getDescription', 'method_getImplementation',
    'method_getName', 'method_getNumberOfArguments',
    'method_getReturnType', 'method_getTypeEncoding', 'method_invoke',
    'method_invoke_stret', 'method_setImplementation',
    'objc_AssociationPolicy', 'objc_addLoadImageFunc',
    'objc_allocateClassPair', 'objc_allocateProtocol',
    'objc_constructInstance', 'objc_copyClassList',
    'objc_copyClassNamesForImage', 'objc_copyImageNames',
    'objc_copyProtocolList', 'objc_destructInstance',
    'objc_disposeClassPair', 'objc_duplicateClass',
    'objc_enumerateClasses', 'objc_enumerationMutation',
    'objc_func_loadImage', 'objc_getAssociatedObject',
    'objc_getClass', 'objc_getClassList', 'objc_getFutureClass',
    'objc_getMetaClass', 'objc_getProtocol', 'objc_getRequiredClass',
    'objc_hook_getClass', 'objc_hook_getImageName',
    'objc_hook_lazyClassNamer', 'objc_loadWeak', 'objc_lookUpClass',
    'objc_msgSend', 'objc_msgSendSuper', 'objc_msgSendSuper_stret',
    'objc_msgSend_stret', 'objc_objectptr_t',
    'objc_property_attribute_t', 'objc_property_t',
    'objc_registerClassPair', 'objc_registerProtocol',
    'objc_removeAssociatedObjects', 'objc_retainedObject',
    'objc_setAssociatedObject', 'objc_setEnumerationMutationHandler',
    'objc_setForwardHandler', 'objc_setHook_getClass',
    'objc_setHook_getImageName', 'objc_setHook_lazyClassNamer',
    'objc_storeWeak', 'objc_unretainedObject',
    'objc_unretainedPointer', 'objc_zone_t', 'object_copy',
    'object_copyFromZone', 'object_dispose', 'object_getClass',
    'object_getClassName', 'object_getIndexedIvars',
    'object_getInstanceVariable', 'object_getIvar', 'object_isClass',
    'object_setClass', 'object_setInstanceVariable',
    'object_setInstanceVariableWithStrongDefault', 'object_setIvar',
    'object_setIvarWithStrongDefault', 'property_copyAttributeList',
    'property_copyAttributeValue', 'property_getAttributes',
    'property_getName', 'protocol_addMethodDescription',
    'protocol_addProperty', 'protocol_addProtocol',
    'protocol_conformsToProtocol',
    'protocol_copyMethodDescriptionList', 'protocol_copyPropertyList',
    'protocol_copyPropertyList2', 'protocol_copyProtocolList',
    'protocol_getMethodDescription', 'protocol_getName',
    'protocol_getProperty', 'protocol_isEqual', 'ptrdiff_t',
    'sel_getName', 'sel_getUid', 'sel_isEqual', 'sel_isMapped',
    'sel_registerName', 'size_t', 'struct__malloc_zone_t',
    'struct_mach_header', 'struct_objc_category', 'struct_objc_class',
    'struct_objc_ivar', 'struct_objc_method',
    'struct_objc_method_description', 'struct_objc_method_list',
    'struct_objc_object', 'struct_objc_property',
    'struct_objc_property_attribute_t', 'struct_objc_selector',
    'struct_objc_super', 'uint8_t']
