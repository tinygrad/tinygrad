# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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





_navi14_ip_offset_HEADER = True # macro
MAX_INSTANCE = 7 # macro
MAX_SEGMENT = 5 # macro
ATHUB_BASE__INST0_SEG0 = 0x00000C00 # macro
ATHUB_BASE__INST0_SEG1 = 0x02408C00 # macro
ATHUB_BASE__INST0_SEG2 = 0 # macro
ATHUB_BASE__INST0_SEG3 = 0 # macro
ATHUB_BASE__INST0_SEG4 = 0 # macro
ATHUB_BASE__INST1_SEG0 = 0 # macro
ATHUB_BASE__INST1_SEG1 = 0 # macro
ATHUB_BASE__INST1_SEG2 = 0 # macro
ATHUB_BASE__INST1_SEG3 = 0 # macro
ATHUB_BASE__INST1_SEG4 = 0 # macro
ATHUB_BASE__INST2_SEG0 = 0 # macro
ATHUB_BASE__INST2_SEG1 = 0 # macro
ATHUB_BASE__INST2_SEG2 = 0 # macro
ATHUB_BASE__INST2_SEG3 = 0 # macro
ATHUB_BASE__INST2_SEG4 = 0 # macro
ATHUB_BASE__INST3_SEG0 = 0 # macro
ATHUB_BASE__INST3_SEG1 = 0 # macro
ATHUB_BASE__INST3_SEG2 = 0 # macro
ATHUB_BASE__INST3_SEG3 = 0 # macro
ATHUB_BASE__INST3_SEG4 = 0 # macro
ATHUB_BASE__INST4_SEG0 = 0 # macro
ATHUB_BASE__INST4_SEG1 = 0 # macro
ATHUB_BASE__INST4_SEG2 = 0 # macro
ATHUB_BASE__INST4_SEG3 = 0 # macro
ATHUB_BASE__INST4_SEG4 = 0 # macro
ATHUB_BASE__INST5_SEG0 = 0 # macro
ATHUB_BASE__INST5_SEG1 = 0 # macro
ATHUB_BASE__INST5_SEG2 = 0 # macro
ATHUB_BASE__INST5_SEG3 = 0 # macro
ATHUB_BASE__INST5_SEG4 = 0 # macro
ATHUB_BASE__INST6_SEG0 = 0 # macro
ATHUB_BASE__INST6_SEG1 = 0 # macro
ATHUB_BASE__INST6_SEG2 = 0 # macro
ATHUB_BASE__INST6_SEG3 = 0 # macro
ATHUB_BASE__INST6_SEG4 = 0 # macro
CLK_BASE__INST0_SEG0 = 0x00016C00 # macro
CLK_BASE__INST0_SEG1 = 0x02401800 # macro
CLK_BASE__INST0_SEG2 = 0 # macro
CLK_BASE__INST0_SEG3 = 0 # macro
CLK_BASE__INST0_SEG4 = 0 # macro
CLK_BASE__INST1_SEG0 = 0x00016E00 # macro
CLK_BASE__INST1_SEG1 = 0x02401C00 # macro
CLK_BASE__INST1_SEG2 = 0 # macro
CLK_BASE__INST1_SEG3 = 0 # macro
CLK_BASE__INST1_SEG4 = 0 # macro
CLK_BASE__INST2_SEG0 = 0x00017000 # macro
CLK_BASE__INST2_SEG1 = 0x02402000 # macro
CLK_BASE__INST2_SEG2 = 0 # macro
CLK_BASE__INST2_SEG3 = 0 # macro
CLK_BASE__INST2_SEG4 = 0 # macro
CLK_BASE__INST3_SEG0 = 0x00017200 # macro
CLK_BASE__INST3_SEG1 = 0x02402400 # macro
CLK_BASE__INST3_SEG2 = 0 # macro
CLK_BASE__INST3_SEG3 = 0 # macro
CLK_BASE__INST3_SEG4 = 0 # macro
CLK_BASE__INST4_SEG0 = 0x0001B000 # macro
CLK_BASE__INST4_SEG1 = 0x0242D800 # macro
CLK_BASE__INST4_SEG2 = 0 # macro
CLK_BASE__INST4_SEG3 = 0 # macro
CLK_BASE__INST4_SEG4 = 0 # macro
CLK_BASE__INST5_SEG0 = 0x00017E00 # macro
CLK_BASE__INST5_SEG1 = 0x0240BC00 # macro
CLK_BASE__INST5_SEG2 = 0 # macro
CLK_BASE__INST5_SEG3 = 0 # macro
CLK_BASE__INST5_SEG4 = 0 # macro
CLK_BASE__INST6_SEG0 = 0 # macro
CLK_BASE__INST6_SEG1 = 0 # macro
CLK_BASE__INST6_SEG2 = 0 # macro
CLK_BASE__INST6_SEG3 = 0 # macro
CLK_BASE__INST6_SEG4 = 0 # macro
DF_BASE__INST0_SEG0 = 0x00007000 # macro
DF_BASE__INST0_SEG1 = 0x0240B800 # macro
DF_BASE__INST0_SEG2 = 0 # macro
DF_BASE__INST0_SEG3 = 0 # macro
DF_BASE__INST0_SEG4 = 0 # macro
DF_BASE__INST1_SEG0 = 0 # macro
DF_BASE__INST1_SEG1 = 0 # macro
DF_BASE__INST1_SEG2 = 0 # macro
DF_BASE__INST1_SEG3 = 0 # macro
DF_BASE__INST1_SEG4 = 0 # macro
DF_BASE__INST2_SEG0 = 0 # macro
DF_BASE__INST2_SEG1 = 0 # macro
DF_BASE__INST2_SEG2 = 0 # macro
DF_BASE__INST2_SEG3 = 0 # macro
DF_BASE__INST2_SEG4 = 0 # macro
DF_BASE__INST3_SEG0 = 0 # macro
DF_BASE__INST3_SEG1 = 0 # macro
DF_BASE__INST3_SEG2 = 0 # macro
DF_BASE__INST3_SEG3 = 0 # macro
DF_BASE__INST3_SEG4 = 0 # macro
DF_BASE__INST4_SEG0 = 0 # macro
DF_BASE__INST4_SEG1 = 0 # macro
DF_BASE__INST4_SEG2 = 0 # macro
DF_BASE__INST4_SEG3 = 0 # macro
DF_BASE__INST4_SEG4 = 0 # macro
DF_BASE__INST5_SEG0 = 0 # macro
DF_BASE__INST5_SEG1 = 0 # macro
DF_BASE__INST5_SEG2 = 0 # macro
DF_BASE__INST5_SEG3 = 0 # macro
DF_BASE__INST5_SEG4 = 0 # macro
DF_BASE__INST6_SEG0 = 0 # macro
DF_BASE__INST6_SEG1 = 0 # macro
DF_BASE__INST6_SEG2 = 0 # macro
DF_BASE__INST6_SEG3 = 0 # macro
DF_BASE__INST6_SEG4 = 0 # macro
DIO_BASE__INST0_SEG0 = 0x02404000 # macro
DIO_BASE__INST0_SEG1 = 0 # macro
DIO_BASE__INST0_SEG2 = 0 # macro
DIO_BASE__INST0_SEG3 = 0 # macro
DIO_BASE__INST0_SEG4 = 0 # macro
DIO_BASE__INST1_SEG0 = 0 # macro
DIO_BASE__INST1_SEG1 = 0 # macro
DIO_BASE__INST1_SEG2 = 0 # macro
DIO_BASE__INST1_SEG3 = 0 # macro
DIO_BASE__INST1_SEG4 = 0 # macro
DIO_BASE__INST2_SEG0 = 0 # macro
DIO_BASE__INST2_SEG1 = 0 # macro
DIO_BASE__INST2_SEG2 = 0 # macro
DIO_BASE__INST2_SEG3 = 0 # macro
DIO_BASE__INST2_SEG4 = 0 # macro
DIO_BASE__INST3_SEG0 = 0 # macro
DIO_BASE__INST3_SEG1 = 0 # macro
DIO_BASE__INST3_SEG2 = 0 # macro
DIO_BASE__INST3_SEG3 = 0 # macro
DIO_BASE__INST3_SEG4 = 0 # macro
DIO_BASE__INST4_SEG0 = 0 # macro
DIO_BASE__INST4_SEG1 = 0 # macro
DIO_BASE__INST4_SEG2 = 0 # macro
DIO_BASE__INST4_SEG3 = 0 # macro
DIO_BASE__INST4_SEG4 = 0 # macro
DIO_BASE__INST5_SEG0 = 0 # macro
DIO_BASE__INST5_SEG1 = 0 # macro
DIO_BASE__INST5_SEG2 = 0 # macro
DIO_BASE__INST5_SEG3 = 0 # macro
DIO_BASE__INST5_SEG4 = 0 # macro
DIO_BASE__INST6_SEG0 = 0 # macro
DIO_BASE__INST6_SEG1 = 0 # macro
DIO_BASE__INST6_SEG2 = 0 # macro
DIO_BASE__INST6_SEG3 = 0 # macro
DIO_BASE__INST6_SEG4 = 0 # macro
DMU_BASE__INST0_SEG0 = 0x00000012 # macro
DMU_BASE__INST0_SEG1 = 0x000000C0 # macro
DMU_BASE__INST0_SEG2 = 0x000034C0 # macro
DMU_BASE__INST0_SEG3 = 0x00009000 # macro
DMU_BASE__INST0_SEG4 = 0x02403C00 # macro
DMU_BASE__INST1_SEG0 = 0 # macro
DMU_BASE__INST1_SEG1 = 0 # macro
DMU_BASE__INST1_SEG2 = 0 # macro
DMU_BASE__INST1_SEG3 = 0 # macro
DMU_BASE__INST1_SEG4 = 0 # macro
DMU_BASE__INST2_SEG0 = 0 # macro
DMU_BASE__INST2_SEG1 = 0 # macro
DMU_BASE__INST2_SEG2 = 0 # macro
DMU_BASE__INST2_SEG3 = 0 # macro
DMU_BASE__INST2_SEG4 = 0 # macro
DMU_BASE__INST3_SEG0 = 0 # macro
DMU_BASE__INST3_SEG1 = 0 # macro
DMU_BASE__INST3_SEG2 = 0 # macro
DMU_BASE__INST3_SEG3 = 0 # macro
DMU_BASE__INST3_SEG4 = 0 # macro
DMU_BASE__INST4_SEG0 = 0 # macro
DMU_BASE__INST4_SEG1 = 0 # macro
DMU_BASE__INST4_SEG2 = 0 # macro
DMU_BASE__INST4_SEG3 = 0 # macro
DMU_BASE__INST4_SEG4 = 0 # macro
DMU_BASE__INST5_SEG0 = 0 # macro
DMU_BASE__INST5_SEG1 = 0 # macro
DMU_BASE__INST5_SEG2 = 0 # macro
DMU_BASE__INST5_SEG3 = 0 # macro
DMU_BASE__INST5_SEG4 = 0 # macro
DMU_BASE__INST6_SEG0 = 0 # macro
DMU_BASE__INST6_SEG1 = 0 # macro
DMU_BASE__INST6_SEG2 = 0 # macro
DMU_BASE__INST6_SEG3 = 0 # macro
DMU_BASE__INST6_SEG4 = 0 # macro
DPCS_BASE__INST0_SEG0 = 0x00000012 # macro
DPCS_BASE__INST0_SEG1 = 0x000000C0 # macro
DPCS_BASE__INST0_SEG2 = 0x000034C0 # macro
DPCS_BASE__INST0_SEG3 = 0x00009000 # macro
DPCS_BASE__INST0_SEG4 = 0x02403C00 # macro
DPCS_BASE__INST1_SEG0 = 0 # macro
DPCS_BASE__INST1_SEG1 = 0 # macro
DPCS_BASE__INST1_SEG2 = 0 # macro
DPCS_BASE__INST1_SEG3 = 0 # macro
DPCS_BASE__INST1_SEG4 = 0 # macro
DPCS_BASE__INST2_SEG0 = 0 # macro
DPCS_BASE__INST2_SEG1 = 0 # macro
DPCS_BASE__INST2_SEG2 = 0 # macro
DPCS_BASE__INST2_SEG3 = 0 # macro
DPCS_BASE__INST2_SEG4 = 0 # macro
DPCS_BASE__INST3_SEG0 = 0 # macro
DPCS_BASE__INST3_SEG1 = 0 # macro
DPCS_BASE__INST3_SEG2 = 0 # macro
DPCS_BASE__INST3_SEG3 = 0 # macro
DPCS_BASE__INST3_SEG4 = 0 # macro
DPCS_BASE__INST4_SEG0 = 0 # macro
DPCS_BASE__INST4_SEG1 = 0 # macro
DPCS_BASE__INST4_SEG2 = 0 # macro
DPCS_BASE__INST4_SEG3 = 0 # macro
DPCS_BASE__INST4_SEG4 = 0 # macro
DPCS_BASE__INST5_SEG0 = 0 # macro
DPCS_BASE__INST5_SEG1 = 0 # macro
DPCS_BASE__INST5_SEG2 = 0 # macro
DPCS_BASE__INST5_SEG3 = 0 # macro
DPCS_BASE__INST5_SEG4 = 0 # macro
DPCS_BASE__INST6_SEG0 = 0 # macro
DPCS_BASE__INST6_SEG1 = 0 # macro
DPCS_BASE__INST6_SEG2 = 0 # macro
DPCS_BASE__INST6_SEG3 = 0 # macro
DPCS_BASE__INST6_SEG4 = 0 # macro
FUSE_BASE__INST0_SEG0 = 0x00017400 # macro
FUSE_BASE__INST0_SEG1 = 0x02401400 # macro
FUSE_BASE__INST0_SEG2 = 0 # macro
FUSE_BASE__INST0_SEG3 = 0 # macro
FUSE_BASE__INST0_SEG4 = 0 # macro
FUSE_BASE__INST1_SEG0 = 0 # macro
FUSE_BASE__INST1_SEG1 = 0 # macro
FUSE_BASE__INST1_SEG2 = 0 # macro
FUSE_BASE__INST1_SEG3 = 0 # macro
FUSE_BASE__INST1_SEG4 = 0 # macro
FUSE_BASE__INST2_SEG0 = 0 # macro
FUSE_BASE__INST2_SEG1 = 0 # macro
FUSE_BASE__INST2_SEG2 = 0 # macro
FUSE_BASE__INST2_SEG3 = 0 # macro
FUSE_BASE__INST2_SEG4 = 0 # macro
FUSE_BASE__INST3_SEG0 = 0 # macro
FUSE_BASE__INST3_SEG1 = 0 # macro
FUSE_BASE__INST3_SEG2 = 0 # macro
FUSE_BASE__INST3_SEG3 = 0 # macro
FUSE_BASE__INST3_SEG4 = 0 # macro
FUSE_BASE__INST4_SEG0 = 0 # macro
FUSE_BASE__INST4_SEG1 = 0 # macro
FUSE_BASE__INST4_SEG2 = 0 # macro
FUSE_BASE__INST4_SEG3 = 0 # macro
FUSE_BASE__INST4_SEG4 = 0 # macro
FUSE_BASE__INST5_SEG0 = 0 # macro
FUSE_BASE__INST5_SEG1 = 0 # macro
FUSE_BASE__INST5_SEG2 = 0 # macro
FUSE_BASE__INST5_SEG3 = 0 # macro
FUSE_BASE__INST5_SEG4 = 0 # macro
FUSE_BASE__INST6_SEG0 = 0 # macro
FUSE_BASE__INST6_SEG1 = 0 # macro
FUSE_BASE__INST6_SEG2 = 0 # macro
FUSE_BASE__INST6_SEG3 = 0 # macro
FUSE_BASE__INST6_SEG4 = 0 # macro
GC_BASE__INST0_SEG0 = 0x00001260 # macro
GC_BASE__INST0_SEG1 = 0x0000A000 # macro
GC_BASE__INST0_SEG2 = 0x02402C00 # macro
GC_BASE__INST0_SEG3 = 0 # macro
GC_BASE__INST0_SEG4 = 0 # macro
GC_BASE__INST1_SEG0 = 0 # macro
GC_BASE__INST1_SEG1 = 0 # macro
GC_BASE__INST1_SEG2 = 0 # macro
GC_BASE__INST1_SEG3 = 0 # macro
GC_BASE__INST1_SEG4 = 0 # macro
GC_BASE__INST2_SEG0 = 0 # macro
GC_BASE__INST2_SEG1 = 0 # macro
GC_BASE__INST2_SEG2 = 0 # macro
GC_BASE__INST2_SEG3 = 0 # macro
GC_BASE__INST2_SEG4 = 0 # macro
GC_BASE__INST3_SEG0 = 0 # macro
GC_BASE__INST3_SEG1 = 0 # macro
GC_BASE__INST3_SEG2 = 0 # macro
GC_BASE__INST3_SEG3 = 0 # macro
GC_BASE__INST3_SEG4 = 0 # macro
GC_BASE__INST4_SEG0 = 0 # macro
GC_BASE__INST4_SEG1 = 0 # macro
GC_BASE__INST4_SEG2 = 0 # macro
GC_BASE__INST4_SEG3 = 0 # macro
GC_BASE__INST4_SEG4 = 0 # macro
GC_BASE__INST5_SEG0 = 0 # macro
GC_BASE__INST5_SEG1 = 0 # macro
GC_BASE__INST5_SEG2 = 0 # macro
GC_BASE__INST5_SEG3 = 0 # macro
GC_BASE__INST5_SEG4 = 0 # macro
GC_BASE__INST6_SEG0 = 0 # macro
GC_BASE__INST6_SEG1 = 0 # macro
GC_BASE__INST6_SEG2 = 0 # macro
GC_BASE__INST6_SEG3 = 0 # macro
GC_BASE__INST6_SEG4 = 0 # macro
HDA_BASE__INST0_SEG0 = 0x004C0000 # macro
HDA_BASE__INST0_SEG1 = 0x02404800 # macro
HDA_BASE__INST0_SEG2 = 0 # macro
HDA_BASE__INST0_SEG3 = 0 # macro
HDA_BASE__INST0_SEG4 = 0 # macro
HDA_BASE__INST1_SEG0 = 0 # macro
HDA_BASE__INST1_SEG1 = 0 # macro
HDA_BASE__INST1_SEG2 = 0 # macro
HDA_BASE__INST1_SEG3 = 0 # macro
HDA_BASE__INST1_SEG4 = 0 # macro
HDA_BASE__INST2_SEG0 = 0 # macro
HDA_BASE__INST2_SEG1 = 0 # macro
HDA_BASE__INST2_SEG2 = 0 # macro
HDA_BASE__INST2_SEG3 = 0 # macro
HDA_BASE__INST2_SEG4 = 0 # macro
HDA_BASE__INST3_SEG0 = 0 # macro
HDA_BASE__INST3_SEG1 = 0 # macro
HDA_BASE__INST3_SEG2 = 0 # macro
HDA_BASE__INST3_SEG3 = 0 # macro
HDA_BASE__INST3_SEG4 = 0 # macro
HDA_BASE__INST4_SEG0 = 0 # macro
HDA_BASE__INST4_SEG1 = 0 # macro
HDA_BASE__INST4_SEG2 = 0 # macro
HDA_BASE__INST4_SEG3 = 0 # macro
HDA_BASE__INST4_SEG4 = 0 # macro
HDA_BASE__INST5_SEG0 = 0 # macro
HDA_BASE__INST5_SEG1 = 0 # macro
HDA_BASE__INST5_SEG2 = 0 # macro
HDA_BASE__INST5_SEG3 = 0 # macro
HDA_BASE__INST5_SEG4 = 0 # macro
HDA_BASE__INST6_SEG0 = 0 # macro
HDA_BASE__INST6_SEG1 = 0 # macro
HDA_BASE__INST6_SEG2 = 0 # macro
HDA_BASE__INST6_SEG3 = 0 # macro
HDA_BASE__INST6_SEG4 = 0 # macro
HDP_BASE__INST0_SEG0 = 0x00000F20 # macro
HDP_BASE__INST0_SEG1 = 0x0240A400 # macro
HDP_BASE__INST0_SEG2 = 0 # macro
HDP_BASE__INST0_SEG3 = 0 # macro
HDP_BASE__INST0_SEG4 = 0 # macro
HDP_BASE__INST1_SEG0 = 0 # macro
HDP_BASE__INST1_SEG1 = 0 # macro
HDP_BASE__INST1_SEG2 = 0 # macro
HDP_BASE__INST1_SEG3 = 0 # macro
HDP_BASE__INST1_SEG4 = 0 # macro
HDP_BASE__INST2_SEG0 = 0 # macro
HDP_BASE__INST2_SEG1 = 0 # macro
HDP_BASE__INST2_SEG2 = 0 # macro
HDP_BASE__INST2_SEG3 = 0 # macro
HDP_BASE__INST2_SEG4 = 0 # macro
HDP_BASE__INST3_SEG0 = 0 # macro
HDP_BASE__INST3_SEG1 = 0 # macro
HDP_BASE__INST3_SEG2 = 0 # macro
HDP_BASE__INST3_SEG3 = 0 # macro
HDP_BASE__INST3_SEG4 = 0 # macro
HDP_BASE__INST4_SEG0 = 0 # macro
HDP_BASE__INST4_SEG1 = 0 # macro
HDP_BASE__INST4_SEG2 = 0 # macro
HDP_BASE__INST4_SEG3 = 0 # macro
HDP_BASE__INST4_SEG4 = 0 # macro
HDP_BASE__INST5_SEG0 = 0 # macro
HDP_BASE__INST5_SEG1 = 0 # macro
HDP_BASE__INST5_SEG2 = 0 # macro
HDP_BASE__INST5_SEG3 = 0 # macro
HDP_BASE__INST5_SEG4 = 0 # macro
HDP_BASE__INST6_SEG0 = 0 # macro
HDP_BASE__INST6_SEG1 = 0 # macro
HDP_BASE__INST6_SEG2 = 0 # macro
HDP_BASE__INST6_SEG3 = 0 # macro
HDP_BASE__INST6_SEG4 = 0 # macro
MMHUB_BASE__INST0_SEG0 = 0x0001A000 # macro
MMHUB_BASE__INST0_SEG1 = 0x02408800 # macro
MMHUB_BASE__INST0_SEG2 = 0 # macro
MMHUB_BASE__INST0_SEG3 = 0 # macro
MMHUB_BASE__INST0_SEG4 = 0 # macro
MMHUB_BASE__INST1_SEG0 = 0 # macro
MMHUB_BASE__INST1_SEG1 = 0 # macro
MMHUB_BASE__INST1_SEG2 = 0 # macro
MMHUB_BASE__INST1_SEG3 = 0 # macro
MMHUB_BASE__INST1_SEG4 = 0 # macro
MMHUB_BASE__INST2_SEG0 = 0 # macro
MMHUB_BASE__INST2_SEG1 = 0 # macro
MMHUB_BASE__INST2_SEG2 = 0 # macro
MMHUB_BASE__INST2_SEG3 = 0 # macro
MMHUB_BASE__INST2_SEG4 = 0 # macro
MMHUB_BASE__INST3_SEG0 = 0 # macro
MMHUB_BASE__INST3_SEG1 = 0 # macro
MMHUB_BASE__INST3_SEG2 = 0 # macro
MMHUB_BASE__INST3_SEG3 = 0 # macro
MMHUB_BASE__INST3_SEG4 = 0 # macro
MMHUB_BASE__INST4_SEG0 = 0 # macro
MMHUB_BASE__INST4_SEG1 = 0 # macro
MMHUB_BASE__INST4_SEG2 = 0 # macro
MMHUB_BASE__INST4_SEG3 = 0 # macro
MMHUB_BASE__INST4_SEG4 = 0 # macro
MMHUB_BASE__INST5_SEG0 = 0 # macro
MMHUB_BASE__INST5_SEG1 = 0 # macro
MMHUB_BASE__INST5_SEG2 = 0 # macro
MMHUB_BASE__INST5_SEG3 = 0 # macro
MMHUB_BASE__INST5_SEG4 = 0 # macro
MMHUB_BASE__INST6_SEG0 = 0 # macro
MMHUB_BASE__INST6_SEG1 = 0 # macro
MMHUB_BASE__INST6_SEG2 = 0 # macro
MMHUB_BASE__INST6_SEG3 = 0 # macro
MMHUB_BASE__INST6_SEG4 = 0 # macro
MP0_BASE__INST0_SEG0 = 0x00016000 # macro
MP0_BASE__INST0_SEG1 = 0x00DC0000 # macro
MP0_BASE__INST0_SEG2 = 0x00E00000 # macro
MP0_BASE__INST0_SEG3 = 0x00E40000 # macro
MP0_BASE__INST0_SEG4 = 0x0243FC00 # macro
MP0_BASE__INST1_SEG0 = 0 # macro
MP0_BASE__INST1_SEG1 = 0 # macro
MP0_BASE__INST1_SEG2 = 0 # macro
MP0_BASE__INST1_SEG3 = 0 # macro
MP0_BASE__INST1_SEG4 = 0 # macro
MP0_BASE__INST2_SEG0 = 0 # macro
MP0_BASE__INST2_SEG1 = 0 # macro
MP0_BASE__INST2_SEG2 = 0 # macro
MP0_BASE__INST2_SEG3 = 0 # macro
MP0_BASE__INST2_SEG4 = 0 # macro
MP0_BASE__INST3_SEG0 = 0 # macro
MP0_BASE__INST3_SEG1 = 0 # macro
MP0_BASE__INST3_SEG2 = 0 # macro
MP0_BASE__INST3_SEG3 = 0 # macro
MP0_BASE__INST3_SEG4 = 0 # macro
MP0_BASE__INST4_SEG0 = 0 # macro
MP0_BASE__INST4_SEG1 = 0 # macro
MP0_BASE__INST4_SEG2 = 0 # macro
MP0_BASE__INST4_SEG3 = 0 # macro
MP0_BASE__INST4_SEG4 = 0 # macro
MP0_BASE__INST5_SEG0 = 0 # macro
MP0_BASE__INST5_SEG1 = 0 # macro
MP0_BASE__INST5_SEG2 = 0 # macro
MP0_BASE__INST5_SEG3 = 0 # macro
MP0_BASE__INST5_SEG4 = 0 # macro
MP0_BASE__INST6_SEG0 = 0 # macro
MP0_BASE__INST6_SEG1 = 0 # macro
MP0_BASE__INST6_SEG2 = 0 # macro
MP0_BASE__INST6_SEG3 = 0 # macro
MP0_BASE__INST6_SEG4 = 0 # macro
MP1_BASE__INST0_SEG0 = 0x00016000 # macro
MP1_BASE__INST0_SEG1 = 0x00DC0000 # macro
MP1_BASE__INST0_SEG2 = 0x00E00000 # macro
MP1_BASE__INST0_SEG3 = 0x00E40000 # macro
MP1_BASE__INST0_SEG4 = 0x0243FC00 # macro
MP1_BASE__INST1_SEG0 = 0 # macro
MP1_BASE__INST1_SEG1 = 0 # macro
MP1_BASE__INST1_SEG2 = 0 # macro
MP1_BASE__INST1_SEG3 = 0 # macro
MP1_BASE__INST1_SEG4 = 0 # macro
MP1_BASE__INST2_SEG0 = 0 # macro
MP1_BASE__INST2_SEG1 = 0 # macro
MP1_BASE__INST2_SEG2 = 0 # macro
MP1_BASE__INST2_SEG3 = 0 # macro
MP1_BASE__INST2_SEG4 = 0 # macro
MP1_BASE__INST3_SEG0 = 0 # macro
MP1_BASE__INST3_SEG1 = 0 # macro
MP1_BASE__INST3_SEG2 = 0 # macro
MP1_BASE__INST3_SEG3 = 0 # macro
MP1_BASE__INST3_SEG4 = 0 # macro
MP1_BASE__INST4_SEG0 = 0 # macro
MP1_BASE__INST4_SEG1 = 0 # macro
MP1_BASE__INST4_SEG2 = 0 # macro
MP1_BASE__INST4_SEG3 = 0 # macro
MP1_BASE__INST4_SEG4 = 0 # macro
MP1_BASE__INST5_SEG0 = 0 # macro
MP1_BASE__INST5_SEG1 = 0 # macro
MP1_BASE__INST5_SEG2 = 0 # macro
MP1_BASE__INST5_SEG3 = 0 # macro
MP1_BASE__INST5_SEG4 = 0 # macro
MP1_BASE__INST6_SEG0 = 0 # macro
MP1_BASE__INST6_SEG1 = 0 # macro
MP1_BASE__INST6_SEG2 = 0 # macro
MP1_BASE__INST6_SEG3 = 0 # macro
MP1_BASE__INST6_SEG4 = 0 # macro
NBIF0_BASE__INST0_SEG0 = 0x00000000 # macro
NBIF0_BASE__INST0_SEG1 = 0x00000014 # macro
NBIF0_BASE__INST0_SEG2 = 0x00000D20 # macro
NBIF0_BASE__INST0_SEG3 = 0x00010400 # macro
NBIF0_BASE__INST0_SEG4 = 0x0241B000 # macro
NBIF0_BASE__INST1_SEG0 = 0 # macro
NBIF0_BASE__INST1_SEG1 = 0 # macro
NBIF0_BASE__INST1_SEG2 = 0 # macro
NBIF0_BASE__INST1_SEG3 = 0 # macro
NBIF0_BASE__INST1_SEG4 = 0 # macro
NBIF0_BASE__INST2_SEG0 = 0 # macro
NBIF0_BASE__INST2_SEG1 = 0 # macro
NBIF0_BASE__INST2_SEG2 = 0 # macro
NBIF0_BASE__INST2_SEG3 = 0 # macro
NBIF0_BASE__INST2_SEG4 = 0 # macro
NBIF0_BASE__INST3_SEG0 = 0 # macro
NBIF0_BASE__INST3_SEG1 = 0 # macro
NBIF0_BASE__INST3_SEG2 = 0 # macro
NBIF0_BASE__INST3_SEG3 = 0 # macro
NBIF0_BASE__INST3_SEG4 = 0 # macro
NBIF0_BASE__INST4_SEG0 = 0 # macro
NBIF0_BASE__INST4_SEG1 = 0 # macro
NBIF0_BASE__INST4_SEG2 = 0 # macro
NBIF0_BASE__INST4_SEG3 = 0 # macro
NBIF0_BASE__INST4_SEG4 = 0 # macro
NBIF0_BASE__INST5_SEG0 = 0 # macro
NBIF0_BASE__INST5_SEG1 = 0 # macro
NBIF0_BASE__INST5_SEG2 = 0 # macro
NBIF0_BASE__INST5_SEG3 = 0 # macro
NBIF0_BASE__INST5_SEG4 = 0 # macro
NBIF0_BASE__INST6_SEG0 = 0 # macro
NBIF0_BASE__INST6_SEG1 = 0 # macro
NBIF0_BASE__INST6_SEG2 = 0 # macro
NBIF0_BASE__INST6_SEG3 = 0 # macro
NBIF0_BASE__INST6_SEG4 = 0 # macro
OSSSYS_BASE__INST0_SEG0 = 0x000010A0 # macro
OSSSYS_BASE__INST0_SEG1 = 0x0240A000 # macro
OSSSYS_BASE__INST0_SEG2 = 0 # macro
OSSSYS_BASE__INST0_SEG3 = 0 # macro
OSSSYS_BASE__INST0_SEG4 = 0 # macro
OSSSYS_BASE__INST1_SEG0 = 0 # macro
OSSSYS_BASE__INST1_SEG1 = 0 # macro
OSSSYS_BASE__INST1_SEG2 = 0 # macro
OSSSYS_BASE__INST1_SEG3 = 0 # macro
OSSSYS_BASE__INST1_SEG4 = 0 # macro
OSSSYS_BASE__INST2_SEG0 = 0 # macro
OSSSYS_BASE__INST2_SEG1 = 0 # macro
OSSSYS_BASE__INST2_SEG2 = 0 # macro
OSSSYS_BASE__INST2_SEG3 = 0 # macro
OSSSYS_BASE__INST2_SEG4 = 0 # macro
OSSSYS_BASE__INST3_SEG0 = 0 # macro
OSSSYS_BASE__INST3_SEG1 = 0 # macro
OSSSYS_BASE__INST3_SEG2 = 0 # macro
OSSSYS_BASE__INST3_SEG3 = 0 # macro
OSSSYS_BASE__INST3_SEG4 = 0 # macro
OSSSYS_BASE__INST4_SEG0 = 0 # macro
OSSSYS_BASE__INST4_SEG1 = 0 # macro
OSSSYS_BASE__INST4_SEG2 = 0 # macro
OSSSYS_BASE__INST4_SEG3 = 0 # macro
OSSSYS_BASE__INST4_SEG4 = 0 # macro
OSSSYS_BASE__INST5_SEG0 = 0 # macro
OSSSYS_BASE__INST5_SEG1 = 0 # macro
OSSSYS_BASE__INST5_SEG2 = 0 # macro
OSSSYS_BASE__INST5_SEG3 = 0 # macro
OSSSYS_BASE__INST5_SEG4 = 0 # macro
OSSSYS_BASE__INST6_SEG0 = 0 # macro
OSSSYS_BASE__INST6_SEG1 = 0 # macro
OSSSYS_BASE__INST6_SEG2 = 0 # macro
OSSSYS_BASE__INST6_SEG3 = 0 # macro
OSSSYS_BASE__INST6_SEG4 = 0 # macro
PCIE0_BASE__INST0_SEG0 = 0x00000000 # macro
PCIE0_BASE__INST0_SEG1 = 0x00000014 # macro
PCIE0_BASE__INST0_SEG2 = 0x00000D20 # macro
PCIE0_BASE__INST0_SEG3 = 0x00010400 # macro
PCIE0_BASE__INST0_SEG4 = 0x0241B000 # macro
PCIE0_BASE__INST1_SEG0 = 0 # macro
PCIE0_BASE__INST1_SEG1 = 0 # macro
PCIE0_BASE__INST1_SEG2 = 0 # macro
PCIE0_BASE__INST1_SEG3 = 0 # macro
PCIE0_BASE__INST1_SEG4 = 0 # macro
PCIE0_BASE__INST2_SEG0 = 0 # macro
PCIE0_BASE__INST2_SEG1 = 0 # macro
PCIE0_BASE__INST2_SEG2 = 0 # macro
PCIE0_BASE__INST2_SEG3 = 0 # macro
PCIE0_BASE__INST2_SEG4 = 0 # macro
PCIE0_BASE__INST3_SEG0 = 0 # macro
PCIE0_BASE__INST3_SEG1 = 0 # macro
PCIE0_BASE__INST3_SEG2 = 0 # macro
PCIE0_BASE__INST3_SEG3 = 0 # macro
PCIE0_BASE__INST3_SEG4 = 0 # macro
PCIE0_BASE__INST4_SEG0 = 0 # macro
PCIE0_BASE__INST4_SEG1 = 0 # macro
PCIE0_BASE__INST4_SEG2 = 0 # macro
PCIE0_BASE__INST4_SEG3 = 0 # macro
PCIE0_BASE__INST4_SEG4 = 0 # macro
PCIE0_BASE__INST5_SEG0 = 0 # macro
PCIE0_BASE__INST5_SEG1 = 0 # macro
PCIE0_BASE__INST5_SEG2 = 0 # macro
PCIE0_BASE__INST5_SEG3 = 0 # macro
PCIE0_BASE__INST5_SEG4 = 0 # macro
PCIE0_BASE__INST6_SEG0 = 0 # macro
PCIE0_BASE__INST6_SEG1 = 0 # macro
PCIE0_BASE__INST6_SEG2 = 0 # macro
PCIE0_BASE__INST6_SEG3 = 0 # macro
PCIE0_BASE__INST6_SEG4 = 0 # macro
SDMA_BASE__INST0_SEG0 = 0x00001260 # macro
SDMA_BASE__INST0_SEG1 = 0x0000A000 # macro
SDMA_BASE__INST0_SEG2 = 0x02402C00 # macro
SDMA_BASE__INST0_SEG3 = 0 # macro
SDMA_BASE__INST0_SEG4 = 0 # macro
SDMA_BASE__INST1_SEG0 = 0x00001260 # macro
SDMA_BASE__INST1_SEG1 = 0x0000A000 # macro
SDMA_BASE__INST1_SEG2 = 0x02402C00 # macro
SDMA_BASE__INST1_SEG3 = 0 # macro
SDMA_BASE__INST1_SEG4 = 0 # macro
SDMA_BASE__INST2_SEG0 = 0 # macro
SDMA_BASE__INST2_SEG1 = 0 # macro
SDMA_BASE__INST2_SEG2 = 0 # macro
SDMA_BASE__INST2_SEG3 = 0 # macro
SDMA_BASE__INST2_SEG4 = 0 # macro
SDMA_BASE__INST3_SEG0 = 0 # macro
SDMA_BASE__INST3_SEG1 = 0 # macro
SDMA_BASE__INST3_SEG2 = 0 # macro
SDMA_BASE__INST3_SEG3 = 0 # macro
SDMA_BASE__INST3_SEG4 = 0 # macro
SDMA_BASE__INST4_SEG0 = 0 # macro
SDMA_BASE__INST4_SEG1 = 0 # macro
SDMA_BASE__INST4_SEG2 = 0 # macro
SDMA_BASE__INST4_SEG3 = 0 # macro
SDMA_BASE__INST4_SEG4 = 0 # macro
SDMA_BASE__INST5_SEG0 = 0 # macro
SDMA_BASE__INST5_SEG1 = 0 # macro
SDMA_BASE__INST5_SEG2 = 0 # macro
SDMA_BASE__INST5_SEG3 = 0 # macro
SDMA_BASE__INST5_SEG4 = 0 # macro
SDMA_BASE__INST6_SEG0 = 0 # macro
SDMA_BASE__INST6_SEG1 = 0 # macro
SDMA_BASE__INST6_SEG2 = 0 # macro
SDMA_BASE__INST6_SEG3 = 0 # macro
SDMA_BASE__INST6_SEG4 = 0 # macro
SMUIO_BASE__INST0_SEG0 = 0x00016800 # macro
SMUIO_BASE__INST0_SEG1 = 0x00016A00 # macro
SMUIO_BASE__INST0_SEG2 = 0x00440000 # macro
SMUIO_BASE__INST0_SEG3 = 0x02401000 # macro
SMUIO_BASE__INST0_SEG4 = 0 # macro
SMUIO_BASE__INST1_SEG0 = 0 # macro
SMUIO_BASE__INST1_SEG1 = 0 # macro
SMUIO_BASE__INST1_SEG2 = 0 # macro
SMUIO_BASE__INST1_SEG3 = 0 # macro
SMUIO_BASE__INST1_SEG4 = 0 # macro
SMUIO_BASE__INST2_SEG0 = 0 # macro
SMUIO_BASE__INST2_SEG1 = 0 # macro
SMUIO_BASE__INST2_SEG2 = 0 # macro
SMUIO_BASE__INST2_SEG3 = 0 # macro
SMUIO_BASE__INST2_SEG4 = 0 # macro
SMUIO_BASE__INST3_SEG0 = 0 # macro
SMUIO_BASE__INST3_SEG1 = 0 # macro
SMUIO_BASE__INST3_SEG2 = 0 # macro
SMUIO_BASE__INST3_SEG3 = 0 # macro
SMUIO_BASE__INST3_SEG4 = 0 # macro
SMUIO_BASE__INST4_SEG0 = 0 # macro
SMUIO_BASE__INST4_SEG1 = 0 # macro
SMUIO_BASE__INST4_SEG2 = 0 # macro
SMUIO_BASE__INST4_SEG3 = 0 # macro
SMUIO_BASE__INST4_SEG4 = 0 # macro
SMUIO_BASE__INST5_SEG0 = 0 # macro
SMUIO_BASE__INST5_SEG1 = 0 # macro
SMUIO_BASE__INST5_SEG2 = 0 # macro
SMUIO_BASE__INST5_SEG3 = 0 # macro
SMUIO_BASE__INST5_SEG4 = 0 # macro
SMUIO_BASE__INST6_SEG0 = 0 # macro
SMUIO_BASE__INST6_SEG1 = 0 # macro
SMUIO_BASE__INST6_SEG2 = 0 # macro
SMUIO_BASE__INST6_SEG3 = 0 # macro
SMUIO_BASE__INST6_SEG4 = 0 # macro
THM_BASE__INST0_SEG0 = 0x00016600 # macro
THM_BASE__INST0_SEG1 = 0x02400C00 # macro
THM_BASE__INST0_SEG2 = 0 # macro
THM_BASE__INST0_SEG3 = 0 # macro
THM_BASE__INST0_SEG4 = 0 # macro
THM_BASE__INST1_SEG0 = 0 # macro
THM_BASE__INST1_SEG1 = 0 # macro
THM_BASE__INST1_SEG2 = 0 # macro
THM_BASE__INST1_SEG3 = 0 # macro
THM_BASE__INST1_SEG4 = 0 # macro
THM_BASE__INST2_SEG0 = 0 # macro
THM_BASE__INST2_SEG1 = 0 # macro
THM_BASE__INST2_SEG2 = 0 # macro
THM_BASE__INST2_SEG3 = 0 # macro
THM_BASE__INST2_SEG4 = 0 # macro
THM_BASE__INST3_SEG0 = 0 # macro
THM_BASE__INST3_SEG1 = 0 # macro
THM_BASE__INST3_SEG2 = 0 # macro
THM_BASE__INST3_SEG3 = 0 # macro
THM_BASE__INST3_SEG4 = 0 # macro
THM_BASE__INST4_SEG0 = 0 # macro
THM_BASE__INST4_SEG1 = 0 # macro
THM_BASE__INST4_SEG2 = 0 # macro
THM_BASE__INST4_SEG3 = 0 # macro
THM_BASE__INST4_SEG4 = 0 # macro
THM_BASE__INST5_SEG0 = 0 # macro
THM_BASE__INST5_SEG1 = 0 # macro
THM_BASE__INST5_SEG2 = 0 # macro
THM_BASE__INST5_SEG3 = 0 # macro
THM_BASE__INST5_SEG4 = 0 # macro
THM_BASE__INST6_SEG0 = 0 # macro
THM_BASE__INST6_SEG1 = 0 # macro
THM_BASE__INST6_SEG2 = 0 # macro
THM_BASE__INST6_SEG3 = 0 # macro
THM_BASE__INST6_SEG4 = 0 # macro
UMC_BASE__INST0_SEG0 = 0x00014000 # macro
UMC_BASE__INST0_SEG1 = 0x02425800 # macro
UMC_BASE__INST0_SEG2 = 0 # macro
UMC_BASE__INST0_SEG3 = 0 # macro
UMC_BASE__INST0_SEG4 = 0 # macro
UMC_BASE__INST1_SEG0 = 0x00054000 # macro
UMC_BASE__INST1_SEG1 = 0x02425C00 # macro
UMC_BASE__INST1_SEG2 = 0 # macro
UMC_BASE__INST1_SEG3 = 0 # macro
UMC_BASE__INST1_SEG4 = 0 # macro
UMC_BASE__INST2_SEG0 = 0x00094000 # macro
UMC_BASE__INST2_SEG1 = 0x02426000 # macro
UMC_BASE__INST2_SEG2 = 0 # macro
UMC_BASE__INST2_SEG3 = 0 # macro
UMC_BASE__INST2_SEG4 = 0 # macro
UMC_BASE__INST3_SEG0 = 0x000D4000 # macro
UMC_BASE__INST3_SEG1 = 0x02426400 # macro
UMC_BASE__INST3_SEG2 = 0 # macro
UMC_BASE__INST3_SEG3 = 0 # macro
UMC_BASE__INST3_SEG4 = 0 # macro
UMC_BASE__INST4_SEG0 = 0 # macro
UMC_BASE__INST4_SEG1 = 0 # macro
UMC_BASE__INST4_SEG2 = 0 # macro
UMC_BASE__INST4_SEG3 = 0 # macro
UMC_BASE__INST4_SEG4 = 0 # macro
UMC_BASE__INST5_SEG0 = 0 # macro
UMC_BASE__INST5_SEG1 = 0 # macro
UMC_BASE__INST5_SEG2 = 0 # macro
UMC_BASE__INST5_SEG3 = 0 # macro
UMC_BASE__INST5_SEG4 = 0 # macro
UMC_BASE__INST6_SEG0 = 0 # macro
UMC_BASE__INST6_SEG1 = 0 # macro
UMC_BASE__INST6_SEG2 = 0 # macro
UMC_BASE__INST6_SEG3 = 0 # macro
UMC_BASE__INST6_SEG4 = 0 # macro
USB0_BASE__INST0_SEG0 = 0x0242A800 # macro
USB0_BASE__INST0_SEG1 = 0x05B00000 # macro
USB0_BASE__INST0_SEG2 = 0 # macro
USB0_BASE__INST0_SEG3 = 0 # macro
USB0_BASE__INST0_SEG4 = 0 # macro
USB0_BASE__INST1_SEG0 = 0 # macro
USB0_BASE__INST1_SEG1 = 0 # macro
USB0_BASE__INST1_SEG2 = 0 # macro
USB0_BASE__INST1_SEG3 = 0 # macro
USB0_BASE__INST1_SEG4 = 0 # macro
USB0_BASE__INST2_SEG0 = 0 # macro
USB0_BASE__INST2_SEG1 = 0 # macro
USB0_BASE__INST2_SEG2 = 0 # macro
USB0_BASE__INST2_SEG3 = 0 # macro
USB0_BASE__INST2_SEG4 = 0 # macro
USB0_BASE__INST3_SEG0 = 0 # macro
USB0_BASE__INST3_SEG1 = 0 # macro
USB0_BASE__INST3_SEG2 = 0 # macro
USB0_BASE__INST3_SEG3 = 0 # macro
USB0_BASE__INST3_SEG4 = 0 # macro
USB0_BASE__INST4_SEG0 = 0 # macro
USB0_BASE__INST4_SEG1 = 0 # macro
USB0_BASE__INST4_SEG2 = 0 # macro
USB0_BASE__INST4_SEG3 = 0 # macro
USB0_BASE__INST4_SEG4 = 0 # macro
USB0_BASE__INST5_SEG0 = 0 # macro
USB0_BASE__INST5_SEG1 = 0 # macro
USB0_BASE__INST5_SEG2 = 0 # macro
USB0_BASE__INST5_SEG3 = 0 # macro
USB0_BASE__INST5_SEG4 = 0 # macro
USB0_BASE__INST6_SEG0 = 0 # macro
USB0_BASE__INST6_SEG1 = 0 # macro
USB0_BASE__INST6_SEG2 = 0 # macro
USB0_BASE__INST6_SEG3 = 0 # macro
USB0_BASE__INST6_SEG4 = 0 # macro
UVD0_BASE__INST0_SEG0 = 0x00007800 # macro
UVD0_BASE__INST0_SEG1 = 0x00007E00 # macro
UVD0_BASE__INST0_SEG2 = 0x02403000 # macro
UVD0_BASE__INST0_SEG3 = 0 # macro
UVD0_BASE__INST0_SEG4 = 0 # macro
UVD0_BASE__INST1_SEG0 = 0 # macro
UVD0_BASE__INST1_SEG1 = 0 # macro
UVD0_BASE__INST1_SEG2 = 0 # macro
UVD0_BASE__INST1_SEG3 = 0 # macro
UVD0_BASE__INST1_SEG4 = 0 # macro
UVD0_BASE__INST2_SEG0 = 0 # macro
UVD0_BASE__INST2_SEG1 = 0 # macro
UVD0_BASE__INST2_SEG2 = 0 # macro
UVD0_BASE__INST2_SEG3 = 0 # macro
UVD0_BASE__INST2_SEG4 = 0 # macro
UVD0_BASE__INST3_SEG0 = 0 # macro
UVD0_BASE__INST3_SEG1 = 0 # macro
UVD0_BASE__INST3_SEG2 = 0 # macro
UVD0_BASE__INST3_SEG3 = 0 # macro
UVD0_BASE__INST3_SEG4 = 0 # macro
UVD0_BASE__INST4_SEG0 = 0 # macro
UVD0_BASE__INST4_SEG1 = 0 # macro
UVD0_BASE__INST4_SEG2 = 0 # macro
UVD0_BASE__INST4_SEG3 = 0 # macro
UVD0_BASE__INST4_SEG4 = 0 # macro
UVD0_BASE__INST5_SEG0 = 0 # macro
UVD0_BASE__INST5_SEG1 = 0 # macro
UVD0_BASE__INST5_SEG2 = 0 # macro
UVD0_BASE__INST5_SEG3 = 0 # macro
UVD0_BASE__INST5_SEG4 = 0 # macro
UVD0_BASE__INST6_SEG0 = 0 # macro
UVD0_BASE__INST6_SEG1 = 0 # macro
UVD0_BASE__INST6_SEG2 = 0 # macro
UVD0_BASE__INST6_SEG3 = 0 # macro
UVD0_BASE__INST6_SEG4 = 0 # macro
class struct_IP_BASE_INSTANCE(Structure):
    pass

struct_IP_BASE_INSTANCE._pack_ = 1 # source:False
struct_IP_BASE_INSTANCE._fields_ = [
    ('segment', ctypes.c_uint32 * 5),
]

class struct_IP_BASE(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('instance', struct_IP_BASE_INSTANCE * 7),
     ]

__maybe_unused = struct_IP_BASE # Variable struct_IP_BASE
ATHUB_BASE = struct_IP_BASE # Variable struct_IP_BASE
CLK_BASE = struct_IP_BASE # Variable struct_IP_BASE
DF_BASE = struct_IP_BASE # Variable struct_IP_BASE
DIO_BASE = struct_IP_BASE # Variable struct_IP_BASE
DMU_BASE = struct_IP_BASE # Variable struct_IP_BASE
DPCS_BASE = struct_IP_BASE # Variable struct_IP_BASE
FUSE_BASE = struct_IP_BASE # Variable struct_IP_BASE
GC_BASE = struct_IP_BASE # Variable struct_IP_BASE
HDA_BASE = struct_IP_BASE # Variable struct_IP_BASE
HDP_BASE = struct_IP_BASE # Variable struct_IP_BASE
MMHUB_BASE = struct_IP_BASE # Variable struct_IP_BASE
MP0_BASE = struct_IP_BASE # Variable struct_IP_BASE
MP1_BASE = struct_IP_BASE # Variable struct_IP_BASE
NBIF0_BASE = struct_IP_BASE # Variable struct_IP_BASE
OSSSYS_BASE = struct_IP_BASE # Variable struct_IP_BASE
PCIE0_BASE = struct_IP_BASE # Variable struct_IP_BASE
SDMA_BASE = struct_IP_BASE # Variable struct_IP_BASE
SMUIO_BASE = struct_IP_BASE # Variable struct_IP_BASE
THM_BASE = struct_IP_BASE # Variable struct_IP_BASE
UMC_BASE = struct_IP_BASE # Variable struct_IP_BASE
USB0_BASE = struct_IP_BASE # Variable struct_IP_BASE
UVD0_BASE = struct_IP_BASE # Variable struct_IP_BASE
__all__ = \
    ['ATHUB_BASE', 'ATHUB_BASE__INST0_SEG0', 'ATHUB_BASE__INST0_SEG1',
    'ATHUB_BASE__INST0_SEG2', 'ATHUB_BASE__INST0_SEG3',
    'ATHUB_BASE__INST0_SEG4', 'ATHUB_BASE__INST1_SEG0',
    'ATHUB_BASE__INST1_SEG1', 'ATHUB_BASE__INST1_SEG2',
    'ATHUB_BASE__INST1_SEG3', 'ATHUB_BASE__INST1_SEG4',
    'ATHUB_BASE__INST2_SEG0', 'ATHUB_BASE__INST2_SEG1',
    'ATHUB_BASE__INST2_SEG2', 'ATHUB_BASE__INST2_SEG3',
    'ATHUB_BASE__INST2_SEG4', 'ATHUB_BASE__INST3_SEG0',
    'ATHUB_BASE__INST3_SEG1', 'ATHUB_BASE__INST3_SEG2',
    'ATHUB_BASE__INST3_SEG3', 'ATHUB_BASE__INST3_SEG4',
    'ATHUB_BASE__INST4_SEG0', 'ATHUB_BASE__INST4_SEG1',
    'ATHUB_BASE__INST4_SEG2', 'ATHUB_BASE__INST4_SEG3',
    'ATHUB_BASE__INST4_SEG4', 'ATHUB_BASE__INST5_SEG0',
    'ATHUB_BASE__INST5_SEG1', 'ATHUB_BASE__INST5_SEG2',
    'ATHUB_BASE__INST5_SEG3', 'ATHUB_BASE__INST5_SEG4',
    'ATHUB_BASE__INST6_SEG0', 'ATHUB_BASE__INST6_SEG1',
    'ATHUB_BASE__INST6_SEG2', 'ATHUB_BASE__INST6_SEG3',
    'ATHUB_BASE__INST6_SEG4', 'CLK_BASE', 'CLK_BASE__INST0_SEG0',
    'CLK_BASE__INST0_SEG1', 'CLK_BASE__INST0_SEG2',
    'CLK_BASE__INST0_SEG3', 'CLK_BASE__INST0_SEG4',
    'CLK_BASE__INST1_SEG0', 'CLK_BASE__INST1_SEG1',
    'CLK_BASE__INST1_SEG2', 'CLK_BASE__INST1_SEG3',
    'CLK_BASE__INST1_SEG4', 'CLK_BASE__INST2_SEG0',
    'CLK_BASE__INST2_SEG1', 'CLK_BASE__INST2_SEG2',
    'CLK_BASE__INST2_SEG3', 'CLK_BASE__INST2_SEG4',
    'CLK_BASE__INST3_SEG0', 'CLK_BASE__INST3_SEG1',
    'CLK_BASE__INST3_SEG2', 'CLK_BASE__INST3_SEG3',
    'CLK_BASE__INST3_SEG4', 'CLK_BASE__INST4_SEG0',
    'CLK_BASE__INST4_SEG1', 'CLK_BASE__INST4_SEG2',
    'CLK_BASE__INST4_SEG3', 'CLK_BASE__INST4_SEG4',
    'CLK_BASE__INST5_SEG0', 'CLK_BASE__INST5_SEG1',
    'CLK_BASE__INST5_SEG2', 'CLK_BASE__INST5_SEG3',
    'CLK_BASE__INST5_SEG4', 'CLK_BASE__INST6_SEG0',
    'CLK_BASE__INST6_SEG1', 'CLK_BASE__INST6_SEG2',
    'CLK_BASE__INST6_SEG3', 'CLK_BASE__INST6_SEG4', 'DF_BASE',
    'DF_BASE__INST0_SEG0', 'DF_BASE__INST0_SEG1',
    'DF_BASE__INST0_SEG2', 'DF_BASE__INST0_SEG3',
    'DF_BASE__INST0_SEG4', 'DF_BASE__INST1_SEG0',
    'DF_BASE__INST1_SEG1', 'DF_BASE__INST1_SEG2',
    'DF_BASE__INST1_SEG3', 'DF_BASE__INST1_SEG4',
    'DF_BASE__INST2_SEG0', 'DF_BASE__INST2_SEG1',
    'DF_BASE__INST2_SEG2', 'DF_BASE__INST2_SEG3',
    'DF_BASE__INST2_SEG4', 'DF_BASE__INST3_SEG0',
    'DF_BASE__INST3_SEG1', 'DF_BASE__INST3_SEG2',
    'DF_BASE__INST3_SEG3', 'DF_BASE__INST3_SEG4',
    'DF_BASE__INST4_SEG0', 'DF_BASE__INST4_SEG1',
    'DF_BASE__INST4_SEG2', 'DF_BASE__INST4_SEG3',
    'DF_BASE__INST4_SEG4', 'DF_BASE__INST5_SEG0',
    'DF_BASE__INST5_SEG1', 'DF_BASE__INST5_SEG2',
    'DF_BASE__INST5_SEG3', 'DF_BASE__INST5_SEG4',
    'DF_BASE__INST6_SEG0', 'DF_BASE__INST6_SEG1',
    'DF_BASE__INST6_SEG2', 'DF_BASE__INST6_SEG3',
    'DF_BASE__INST6_SEG4', 'DIO_BASE', 'DIO_BASE__INST0_SEG0',
    'DIO_BASE__INST0_SEG1', 'DIO_BASE__INST0_SEG2',
    'DIO_BASE__INST0_SEG3', 'DIO_BASE__INST0_SEG4',
    'DIO_BASE__INST1_SEG0', 'DIO_BASE__INST1_SEG1',
    'DIO_BASE__INST1_SEG2', 'DIO_BASE__INST1_SEG3',
    'DIO_BASE__INST1_SEG4', 'DIO_BASE__INST2_SEG0',
    'DIO_BASE__INST2_SEG1', 'DIO_BASE__INST2_SEG2',
    'DIO_BASE__INST2_SEG3', 'DIO_BASE__INST2_SEG4',
    'DIO_BASE__INST3_SEG0', 'DIO_BASE__INST3_SEG1',
    'DIO_BASE__INST3_SEG2', 'DIO_BASE__INST3_SEG3',
    'DIO_BASE__INST3_SEG4', 'DIO_BASE__INST4_SEG0',
    'DIO_BASE__INST4_SEG1', 'DIO_BASE__INST4_SEG2',
    'DIO_BASE__INST4_SEG3', 'DIO_BASE__INST4_SEG4',
    'DIO_BASE__INST5_SEG0', 'DIO_BASE__INST5_SEG1',
    'DIO_BASE__INST5_SEG2', 'DIO_BASE__INST5_SEG3',
    'DIO_BASE__INST5_SEG4', 'DIO_BASE__INST6_SEG0',
    'DIO_BASE__INST6_SEG1', 'DIO_BASE__INST6_SEG2',
    'DIO_BASE__INST6_SEG3', 'DIO_BASE__INST6_SEG4', 'DMU_BASE',
    'DMU_BASE__INST0_SEG0', 'DMU_BASE__INST0_SEG1',
    'DMU_BASE__INST0_SEG2', 'DMU_BASE__INST0_SEG3',
    'DMU_BASE__INST0_SEG4', 'DMU_BASE__INST1_SEG0',
    'DMU_BASE__INST1_SEG1', 'DMU_BASE__INST1_SEG2',
    'DMU_BASE__INST1_SEG3', 'DMU_BASE__INST1_SEG4',
    'DMU_BASE__INST2_SEG0', 'DMU_BASE__INST2_SEG1',
    'DMU_BASE__INST2_SEG2', 'DMU_BASE__INST2_SEG3',
    'DMU_BASE__INST2_SEG4', 'DMU_BASE__INST3_SEG0',
    'DMU_BASE__INST3_SEG1', 'DMU_BASE__INST3_SEG2',
    'DMU_BASE__INST3_SEG3', 'DMU_BASE__INST3_SEG4',
    'DMU_BASE__INST4_SEG0', 'DMU_BASE__INST4_SEG1',
    'DMU_BASE__INST4_SEG2', 'DMU_BASE__INST4_SEG3',
    'DMU_BASE__INST4_SEG4', 'DMU_BASE__INST5_SEG0',
    'DMU_BASE__INST5_SEG1', 'DMU_BASE__INST5_SEG2',
    'DMU_BASE__INST5_SEG3', 'DMU_BASE__INST5_SEG4',
    'DMU_BASE__INST6_SEG0', 'DMU_BASE__INST6_SEG1',
    'DMU_BASE__INST6_SEG2', 'DMU_BASE__INST6_SEG3',
    'DMU_BASE__INST6_SEG4', 'DPCS_BASE', 'DPCS_BASE__INST0_SEG0',
    'DPCS_BASE__INST0_SEG1', 'DPCS_BASE__INST0_SEG2',
    'DPCS_BASE__INST0_SEG3', 'DPCS_BASE__INST0_SEG4',
    'DPCS_BASE__INST1_SEG0', 'DPCS_BASE__INST1_SEG1',
    'DPCS_BASE__INST1_SEG2', 'DPCS_BASE__INST1_SEG3',
    'DPCS_BASE__INST1_SEG4', 'DPCS_BASE__INST2_SEG0',
    'DPCS_BASE__INST2_SEG1', 'DPCS_BASE__INST2_SEG2',
    'DPCS_BASE__INST2_SEG3', 'DPCS_BASE__INST2_SEG4',
    'DPCS_BASE__INST3_SEG0', 'DPCS_BASE__INST3_SEG1',
    'DPCS_BASE__INST3_SEG2', 'DPCS_BASE__INST3_SEG3',
    'DPCS_BASE__INST3_SEG4', 'DPCS_BASE__INST4_SEG0',
    'DPCS_BASE__INST4_SEG1', 'DPCS_BASE__INST4_SEG2',
    'DPCS_BASE__INST4_SEG3', 'DPCS_BASE__INST4_SEG4',
    'DPCS_BASE__INST5_SEG0', 'DPCS_BASE__INST5_SEG1',
    'DPCS_BASE__INST5_SEG2', 'DPCS_BASE__INST5_SEG3',
    'DPCS_BASE__INST5_SEG4', 'DPCS_BASE__INST6_SEG0',
    'DPCS_BASE__INST6_SEG1', 'DPCS_BASE__INST6_SEG2',
    'DPCS_BASE__INST6_SEG3', 'DPCS_BASE__INST6_SEG4', 'FUSE_BASE',
    'FUSE_BASE__INST0_SEG0', 'FUSE_BASE__INST0_SEG1',
    'FUSE_BASE__INST0_SEG2', 'FUSE_BASE__INST0_SEG3',
    'FUSE_BASE__INST0_SEG4', 'FUSE_BASE__INST1_SEG0',
    'FUSE_BASE__INST1_SEG1', 'FUSE_BASE__INST1_SEG2',
    'FUSE_BASE__INST1_SEG3', 'FUSE_BASE__INST1_SEG4',
    'FUSE_BASE__INST2_SEG0', 'FUSE_BASE__INST2_SEG1',
    'FUSE_BASE__INST2_SEG2', 'FUSE_BASE__INST2_SEG3',
    'FUSE_BASE__INST2_SEG4', 'FUSE_BASE__INST3_SEG0',
    'FUSE_BASE__INST3_SEG1', 'FUSE_BASE__INST3_SEG2',
    'FUSE_BASE__INST3_SEG3', 'FUSE_BASE__INST3_SEG4',
    'FUSE_BASE__INST4_SEG0', 'FUSE_BASE__INST4_SEG1',
    'FUSE_BASE__INST4_SEG2', 'FUSE_BASE__INST4_SEG3',
    'FUSE_BASE__INST4_SEG4', 'FUSE_BASE__INST5_SEG0',
    'FUSE_BASE__INST5_SEG1', 'FUSE_BASE__INST5_SEG2',
    'FUSE_BASE__INST5_SEG3', 'FUSE_BASE__INST5_SEG4',
    'FUSE_BASE__INST6_SEG0', 'FUSE_BASE__INST6_SEG1',
    'FUSE_BASE__INST6_SEG2', 'FUSE_BASE__INST6_SEG3',
    'FUSE_BASE__INST6_SEG4', 'GC_BASE', 'GC_BASE__INST0_SEG0',
    'GC_BASE__INST0_SEG1', 'GC_BASE__INST0_SEG2',
    'GC_BASE__INST0_SEG3', 'GC_BASE__INST0_SEG4',
    'GC_BASE__INST1_SEG0', 'GC_BASE__INST1_SEG1',
    'GC_BASE__INST1_SEG2', 'GC_BASE__INST1_SEG3',
    'GC_BASE__INST1_SEG4', 'GC_BASE__INST2_SEG0',
    'GC_BASE__INST2_SEG1', 'GC_BASE__INST2_SEG2',
    'GC_BASE__INST2_SEG3', 'GC_BASE__INST2_SEG4',
    'GC_BASE__INST3_SEG0', 'GC_BASE__INST3_SEG1',
    'GC_BASE__INST3_SEG2', 'GC_BASE__INST3_SEG3',
    'GC_BASE__INST3_SEG4', 'GC_BASE__INST4_SEG0',
    'GC_BASE__INST4_SEG1', 'GC_BASE__INST4_SEG2',
    'GC_BASE__INST4_SEG3', 'GC_BASE__INST4_SEG4',
    'GC_BASE__INST5_SEG0', 'GC_BASE__INST5_SEG1',
    'GC_BASE__INST5_SEG2', 'GC_BASE__INST5_SEG3',
    'GC_BASE__INST5_SEG4', 'GC_BASE__INST6_SEG0',
    'GC_BASE__INST6_SEG1', 'GC_BASE__INST6_SEG2',
    'GC_BASE__INST6_SEG3', 'GC_BASE__INST6_SEG4', 'HDA_BASE',
    'HDA_BASE__INST0_SEG0', 'HDA_BASE__INST0_SEG1',
    'HDA_BASE__INST0_SEG2', 'HDA_BASE__INST0_SEG3',
    'HDA_BASE__INST0_SEG4', 'HDA_BASE__INST1_SEG0',
    'HDA_BASE__INST1_SEG1', 'HDA_BASE__INST1_SEG2',
    'HDA_BASE__INST1_SEG3', 'HDA_BASE__INST1_SEG4',
    'HDA_BASE__INST2_SEG0', 'HDA_BASE__INST2_SEG1',
    'HDA_BASE__INST2_SEG2', 'HDA_BASE__INST2_SEG3',
    'HDA_BASE__INST2_SEG4', 'HDA_BASE__INST3_SEG0',
    'HDA_BASE__INST3_SEG1', 'HDA_BASE__INST3_SEG2',
    'HDA_BASE__INST3_SEG3', 'HDA_BASE__INST3_SEG4',
    'HDA_BASE__INST4_SEG0', 'HDA_BASE__INST4_SEG1',
    'HDA_BASE__INST4_SEG2', 'HDA_BASE__INST4_SEG3',
    'HDA_BASE__INST4_SEG4', 'HDA_BASE__INST5_SEG0',
    'HDA_BASE__INST5_SEG1', 'HDA_BASE__INST5_SEG2',
    'HDA_BASE__INST5_SEG3', 'HDA_BASE__INST5_SEG4',
    'HDA_BASE__INST6_SEG0', 'HDA_BASE__INST6_SEG1',
    'HDA_BASE__INST6_SEG2', 'HDA_BASE__INST6_SEG3',
    'HDA_BASE__INST6_SEG4', 'HDP_BASE', 'HDP_BASE__INST0_SEG0',
    'HDP_BASE__INST0_SEG1', 'HDP_BASE__INST0_SEG2',
    'HDP_BASE__INST0_SEG3', 'HDP_BASE__INST0_SEG4',
    'HDP_BASE__INST1_SEG0', 'HDP_BASE__INST1_SEG1',
    'HDP_BASE__INST1_SEG2', 'HDP_BASE__INST1_SEG3',
    'HDP_BASE__INST1_SEG4', 'HDP_BASE__INST2_SEG0',
    'HDP_BASE__INST2_SEG1', 'HDP_BASE__INST2_SEG2',
    'HDP_BASE__INST2_SEG3', 'HDP_BASE__INST2_SEG4',
    'HDP_BASE__INST3_SEG0', 'HDP_BASE__INST3_SEG1',
    'HDP_BASE__INST3_SEG2', 'HDP_BASE__INST3_SEG3',
    'HDP_BASE__INST3_SEG4', 'HDP_BASE__INST4_SEG0',
    'HDP_BASE__INST4_SEG1', 'HDP_BASE__INST4_SEG2',
    'HDP_BASE__INST4_SEG3', 'HDP_BASE__INST4_SEG4',
    'HDP_BASE__INST5_SEG0', 'HDP_BASE__INST5_SEG1',
    'HDP_BASE__INST5_SEG2', 'HDP_BASE__INST5_SEG3',
    'HDP_BASE__INST5_SEG4', 'HDP_BASE__INST6_SEG0',
    'HDP_BASE__INST6_SEG1', 'HDP_BASE__INST6_SEG2',
    'HDP_BASE__INST6_SEG3', 'HDP_BASE__INST6_SEG4', 'MAX_INSTANCE',
    'MAX_SEGMENT', 'MMHUB_BASE', 'MMHUB_BASE__INST0_SEG0',
    'MMHUB_BASE__INST0_SEG1', 'MMHUB_BASE__INST0_SEG2',
    'MMHUB_BASE__INST0_SEG3', 'MMHUB_BASE__INST0_SEG4',
    'MMHUB_BASE__INST1_SEG0', 'MMHUB_BASE__INST1_SEG1',
    'MMHUB_BASE__INST1_SEG2', 'MMHUB_BASE__INST1_SEG3',
    'MMHUB_BASE__INST1_SEG4', 'MMHUB_BASE__INST2_SEG0',
    'MMHUB_BASE__INST2_SEG1', 'MMHUB_BASE__INST2_SEG2',
    'MMHUB_BASE__INST2_SEG3', 'MMHUB_BASE__INST2_SEG4',
    'MMHUB_BASE__INST3_SEG0', 'MMHUB_BASE__INST3_SEG1',
    'MMHUB_BASE__INST3_SEG2', 'MMHUB_BASE__INST3_SEG3',
    'MMHUB_BASE__INST3_SEG4', 'MMHUB_BASE__INST4_SEG0',
    'MMHUB_BASE__INST4_SEG1', 'MMHUB_BASE__INST4_SEG2',
    'MMHUB_BASE__INST4_SEG3', 'MMHUB_BASE__INST4_SEG4',
    'MMHUB_BASE__INST5_SEG0', 'MMHUB_BASE__INST5_SEG1',
    'MMHUB_BASE__INST5_SEG2', 'MMHUB_BASE__INST5_SEG3',
    'MMHUB_BASE__INST5_SEG4', 'MMHUB_BASE__INST6_SEG0',
    'MMHUB_BASE__INST6_SEG1', 'MMHUB_BASE__INST6_SEG2',
    'MMHUB_BASE__INST6_SEG3', 'MMHUB_BASE__INST6_SEG4', 'MP0_BASE',
    'MP0_BASE__INST0_SEG0', 'MP0_BASE__INST0_SEG1',
    'MP0_BASE__INST0_SEG2', 'MP0_BASE__INST0_SEG3',
    'MP0_BASE__INST0_SEG4', 'MP0_BASE__INST1_SEG0',
    'MP0_BASE__INST1_SEG1', 'MP0_BASE__INST1_SEG2',
    'MP0_BASE__INST1_SEG3', 'MP0_BASE__INST1_SEG4',
    'MP0_BASE__INST2_SEG0', 'MP0_BASE__INST2_SEG1',
    'MP0_BASE__INST2_SEG2', 'MP0_BASE__INST2_SEG3',
    'MP0_BASE__INST2_SEG4', 'MP0_BASE__INST3_SEG0',
    'MP0_BASE__INST3_SEG1', 'MP0_BASE__INST3_SEG2',
    'MP0_BASE__INST3_SEG3', 'MP0_BASE__INST3_SEG4',
    'MP0_BASE__INST4_SEG0', 'MP0_BASE__INST4_SEG1',
    'MP0_BASE__INST4_SEG2', 'MP0_BASE__INST4_SEG3',
    'MP0_BASE__INST4_SEG4', 'MP0_BASE__INST5_SEG0',
    'MP0_BASE__INST5_SEG1', 'MP0_BASE__INST5_SEG2',
    'MP0_BASE__INST5_SEG3', 'MP0_BASE__INST5_SEG4',
    'MP0_BASE__INST6_SEG0', 'MP0_BASE__INST6_SEG1',
    'MP0_BASE__INST6_SEG2', 'MP0_BASE__INST6_SEG3',
    'MP0_BASE__INST6_SEG4', 'MP1_BASE', 'MP1_BASE__INST0_SEG0',
    'MP1_BASE__INST0_SEG1', 'MP1_BASE__INST0_SEG2',
    'MP1_BASE__INST0_SEG3', 'MP1_BASE__INST0_SEG4',
    'MP1_BASE__INST1_SEG0', 'MP1_BASE__INST1_SEG1',
    'MP1_BASE__INST1_SEG2', 'MP1_BASE__INST1_SEG3',
    'MP1_BASE__INST1_SEG4', 'MP1_BASE__INST2_SEG0',
    'MP1_BASE__INST2_SEG1', 'MP1_BASE__INST2_SEG2',
    'MP1_BASE__INST2_SEG3', 'MP1_BASE__INST2_SEG4',
    'MP1_BASE__INST3_SEG0', 'MP1_BASE__INST3_SEG1',
    'MP1_BASE__INST3_SEG2', 'MP1_BASE__INST3_SEG3',
    'MP1_BASE__INST3_SEG4', 'MP1_BASE__INST4_SEG0',
    'MP1_BASE__INST4_SEG1', 'MP1_BASE__INST4_SEG2',
    'MP1_BASE__INST4_SEG3', 'MP1_BASE__INST4_SEG4',
    'MP1_BASE__INST5_SEG0', 'MP1_BASE__INST5_SEG1',
    'MP1_BASE__INST5_SEG2', 'MP1_BASE__INST5_SEG3',
    'MP1_BASE__INST5_SEG4', 'MP1_BASE__INST6_SEG0',
    'MP1_BASE__INST6_SEG1', 'MP1_BASE__INST6_SEG2',
    'MP1_BASE__INST6_SEG3', 'MP1_BASE__INST6_SEG4', 'NBIF0_BASE',
    'NBIF0_BASE__INST0_SEG0', 'NBIF0_BASE__INST0_SEG1',
    'NBIF0_BASE__INST0_SEG2', 'NBIF0_BASE__INST0_SEG3',
    'NBIF0_BASE__INST0_SEG4', 'NBIF0_BASE__INST1_SEG0',
    'NBIF0_BASE__INST1_SEG1', 'NBIF0_BASE__INST1_SEG2',
    'NBIF0_BASE__INST1_SEG3', 'NBIF0_BASE__INST1_SEG4',
    'NBIF0_BASE__INST2_SEG0', 'NBIF0_BASE__INST2_SEG1',
    'NBIF0_BASE__INST2_SEG2', 'NBIF0_BASE__INST2_SEG3',
    'NBIF0_BASE__INST2_SEG4', 'NBIF0_BASE__INST3_SEG0',
    'NBIF0_BASE__INST3_SEG1', 'NBIF0_BASE__INST3_SEG2',
    'NBIF0_BASE__INST3_SEG3', 'NBIF0_BASE__INST3_SEG4',
    'NBIF0_BASE__INST4_SEG0', 'NBIF0_BASE__INST4_SEG1',
    'NBIF0_BASE__INST4_SEG2', 'NBIF0_BASE__INST4_SEG3',
    'NBIF0_BASE__INST4_SEG4', 'NBIF0_BASE__INST5_SEG0',
    'NBIF0_BASE__INST5_SEG1', 'NBIF0_BASE__INST5_SEG2',
    'NBIF0_BASE__INST5_SEG3', 'NBIF0_BASE__INST5_SEG4',
    'NBIF0_BASE__INST6_SEG0', 'NBIF0_BASE__INST6_SEG1',
    'NBIF0_BASE__INST6_SEG2', 'NBIF0_BASE__INST6_SEG3',
    'NBIF0_BASE__INST6_SEG4', 'OSSSYS_BASE',
    'OSSSYS_BASE__INST0_SEG0', 'OSSSYS_BASE__INST0_SEG1',
    'OSSSYS_BASE__INST0_SEG2', 'OSSSYS_BASE__INST0_SEG3',
    'OSSSYS_BASE__INST0_SEG4', 'OSSSYS_BASE__INST1_SEG0',
    'OSSSYS_BASE__INST1_SEG1', 'OSSSYS_BASE__INST1_SEG2',
    'OSSSYS_BASE__INST1_SEG3', 'OSSSYS_BASE__INST1_SEG4',
    'OSSSYS_BASE__INST2_SEG0', 'OSSSYS_BASE__INST2_SEG1',
    'OSSSYS_BASE__INST2_SEG2', 'OSSSYS_BASE__INST2_SEG3',
    'OSSSYS_BASE__INST2_SEG4', 'OSSSYS_BASE__INST3_SEG0',
    'OSSSYS_BASE__INST3_SEG1', 'OSSSYS_BASE__INST3_SEG2',
    'OSSSYS_BASE__INST3_SEG3', 'OSSSYS_BASE__INST3_SEG4',
    'OSSSYS_BASE__INST4_SEG0', 'OSSSYS_BASE__INST4_SEG1',
    'OSSSYS_BASE__INST4_SEG2', 'OSSSYS_BASE__INST4_SEG3',
    'OSSSYS_BASE__INST4_SEG4', 'OSSSYS_BASE__INST5_SEG0',
    'OSSSYS_BASE__INST5_SEG1', 'OSSSYS_BASE__INST5_SEG2',
    'OSSSYS_BASE__INST5_SEG3', 'OSSSYS_BASE__INST5_SEG4',
    'OSSSYS_BASE__INST6_SEG0', 'OSSSYS_BASE__INST6_SEG1',
    'OSSSYS_BASE__INST6_SEG2', 'OSSSYS_BASE__INST6_SEG3',
    'OSSSYS_BASE__INST6_SEG4', 'PCIE0_BASE', 'PCIE0_BASE__INST0_SEG0',
    'PCIE0_BASE__INST0_SEG1', 'PCIE0_BASE__INST0_SEG2',
    'PCIE0_BASE__INST0_SEG3', 'PCIE0_BASE__INST0_SEG4',
    'PCIE0_BASE__INST1_SEG0', 'PCIE0_BASE__INST1_SEG1',
    'PCIE0_BASE__INST1_SEG2', 'PCIE0_BASE__INST1_SEG3',
    'PCIE0_BASE__INST1_SEG4', 'PCIE0_BASE__INST2_SEG0',
    'PCIE0_BASE__INST2_SEG1', 'PCIE0_BASE__INST2_SEG2',
    'PCIE0_BASE__INST2_SEG3', 'PCIE0_BASE__INST2_SEG4',
    'PCIE0_BASE__INST3_SEG0', 'PCIE0_BASE__INST3_SEG1',
    'PCIE0_BASE__INST3_SEG2', 'PCIE0_BASE__INST3_SEG3',
    'PCIE0_BASE__INST3_SEG4', 'PCIE0_BASE__INST4_SEG0',
    'PCIE0_BASE__INST4_SEG1', 'PCIE0_BASE__INST4_SEG2',
    'PCIE0_BASE__INST4_SEG3', 'PCIE0_BASE__INST4_SEG4',
    'PCIE0_BASE__INST5_SEG0', 'PCIE0_BASE__INST5_SEG1',
    'PCIE0_BASE__INST5_SEG2', 'PCIE0_BASE__INST5_SEG3',
    'PCIE0_BASE__INST5_SEG4', 'PCIE0_BASE__INST6_SEG0',
    'PCIE0_BASE__INST6_SEG1', 'PCIE0_BASE__INST6_SEG2',
    'PCIE0_BASE__INST6_SEG3', 'PCIE0_BASE__INST6_SEG4', 'SDMA_BASE',
    'SDMA_BASE__INST0_SEG0', 'SDMA_BASE__INST0_SEG1',
    'SDMA_BASE__INST0_SEG2', 'SDMA_BASE__INST0_SEG3',
    'SDMA_BASE__INST0_SEG4', 'SDMA_BASE__INST1_SEG0',
    'SDMA_BASE__INST1_SEG1', 'SDMA_BASE__INST1_SEG2',
    'SDMA_BASE__INST1_SEG3', 'SDMA_BASE__INST1_SEG4',
    'SDMA_BASE__INST2_SEG0', 'SDMA_BASE__INST2_SEG1',
    'SDMA_BASE__INST2_SEG2', 'SDMA_BASE__INST2_SEG3',
    'SDMA_BASE__INST2_SEG4', 'SDMA_BASE__INST3_SEG0',
    'SDMA_BASE__INST3_SEG1', 'SDMA_BASE__INST3_SEG2',
    'SDMA_BASE__INST3_SEG3', 'SDMA_BASE__INST3_SEG4',
    'SDMA_BASE__INST4_SEG0', 'SDMA_BASE__INST4_SEG1',
    'SDMA_BASE__INST4_SEG2', 'SDMA_BASE__INST4_SEG3',
    'SDMA_BASE__INST4_SEG4', 'SDMA_BASE__INST5_SEG0',
    'SDMA_BASE__INST5_SEG1', 'SDMA_BASE__INST5_SEG2',
    'SDMA_BASE__INST5_SEG3', 'SDMA_BASE__INST5_SEG4',
    'SDMA_BASE__INST6_SEG0', 'SDMA_BASE__INST6_SEG1',
    'SDMA_BASE__INST6_SEG2', 'SDMA_BASE__INST6_SEG3',
    'SDMA_BASE__INST6_SEG4', 'SMUIO_BASE', 'SMUIO_BASE__INST0_SEG0',
    'SMUIO_BASE__INST0_SEG1', 'SMUIO_BASE__INST0_SEG2',
    'SMUIO_BASE__INST0_SEG3', 'SMUIO_BASE__INST0_SEG4',
    'SMUIO_BASE__INST1_SEG0', 'SMUIO_BASE__INST1_SEG1',
    'SMUIO_BASE__INST1_SEG2', 'SMUIO_BASE__INST1_SEG3',
    'SMUIO_BASE__INST1_SEG4', 'SMUIO_BASE__INST2_SEG0',
    'SMUIO_BASE__INST2_SEG1', 'SMUIO_BASE__INST2_SEG2',
    'SMUIO_BASE__INST2_SEG3', 'SMUIO_BASE__INST2_SEG4',
    'SMUIO_BASE__INST3_SEG0', 'SMUIO_BASE__INST3_SEG1',
    'SMUIO_BASE__INST3_SEG2', 'SMUIO_BASE__INST3_SEG3',
    'SMUIO_BASE__INST3_SEG4', 'SMUIO_BASE__INST4_SEG0',
    'SMUIO_BASE__INST4_SEG1', 'SMUIO_BASE__INST4_SEG2',
    'SMUIO_BASE__INST4_SEG3', 'SMUIO_BASE__INST4_SEG4',
    'SMUIO_BASE__INST5_SEG0', 'SMUIO_BASE__INST5_SEG1',
    'SMUIO_BASE__INST5_SEG2', 'SMUIO_BASE__INST5_SEG3',
    'SMUIO_BASE__INST5_SEG4', 'SMUIO_BASE__INST6_SEG0',
    'SMUIO_BASE__INST6_SEG1', 'SMUIO_BASE__INST6_SEG2',
    'SMUIO_BASE__INST6_SEG3', 'SMUIO_BASE__INST6_SEG4', 'THM_BASE',
    'THM_BASE__INST0_SEG0', 'THM_BASE__INST0_SEG1',
    'THM_BASE__INST0_SEG2', 'THM_BASE__INST0_SEG3',
    'THM_BASE__INST0_SEG4', 'THM_BASE__INST1_SEG0',
    'THM_BASE__INST1_SEG1', 'THM_BASE__INST1_SEG2',
    'THM_BASE__INST1_SEG3', 'THM_BASE__INST1_SEG4',
    'THM_BASE__INST2_SEG0', 'THM_BASE__INST2_SEG1',
    'THM_BASE__INST2_SEG2', 'THM_BASE__INST2_SEG3',
    'THM_BASE__INST2_SEG4', 'THM_BASE__INST3_SEG0',
    'THM_BASE__INST3_SEG1', 'THM_BASE__INST3_SEG2',
    'THM_BASE__INST3_SEG3', 'THM_BASE__INST3_SEG4',
    'THM_BASE__INST4_SEG0', 'THM_BASE__INST4_SEG1',
    'THM_BASE__INST4_SEG2', 'THM_BASE__INST4_SEG3',
    'THM_BASE__INST4_SEG4', 'THM_BASE__INST5_SEG0',
    'THM_BASE__INST5_SEG1', 'THM_BASE__INST5_SEG2',
    'THM_BASE__INST5_SEG3', 'THM_BASE__INST5_SEG4',
    'THM_BASE__INST6_SEG0', 'THM_BASE__INST6_SEG1',
    'THM_BASE__INST6_SEG2', 'THM_BASE__INST6_SEG3',
    'THM_BASE__INST6_SEG4', 'UMC_BASE', 'UMC_BASE__INST0_SEG0',
    'UMC_BASE__INST0_SEG1', 'UMC_BASE__INST0_SEG2',
    'UMC_BASE__INST0_SEG3', 'UMC_BASE__INST0_SEG4',
    'UMC_BASE__INST1_SEG0', 'UMC_BASE__INST1_SEG1',
    'UMC_BASE__INST1_SEG2', 'UMC_BASE__INST1_SEG3',
    'UMC_BASE__INST1_SEG4', 'UMC_BASE__INST2_SEG0',
    'UMC_BASE__INST2_SEG1', 'UMC_BASE__INST2_SEG2',
    'UMC_BASE__INST2_SEG3', 'UMC_BASE__INST2_SEG4',
    'UMC_BASE__INST3_SEG0', 'UMC_BASE__INST3_SEG1',
    'UMC_BASE__INST3_SEG2', 'UMC_BASE__INST3_SEG3',
    'UMC_BASE__INST3_SEG4', 'UMC_BASE__INST4_SEG0',
    'UMC_BASE__INST4_SEG1', 'UMC_BASE__INST4_SEG2',
    'UMC_BASE__INST4_SEG3', 'UMC_BASE__INST4_SEG4',
    'UMC_BASE__INST5_SEG0', 'UMC_BASE__INST5_SEG1',
    'UMC_BASE__INST5_SEG2', 'UMC_BASE__INST5_SEG3',
    'UMC_BASE__INST5_SEG4', 'UMC_BASE__INST6_SEG0',
    'UMC_BASE__INST6_SEG1', 'UMC_BASE__INST6_SEG2',
    'UMC_BASE__INST6_SEG3', 'UMC_BASE__INST6_SEG4', 'USB0_BASE',
    'USB0_BASE__INST0_SEG0', 'USB0_BASE__INST0_SEG1',
    'USB0_BASE__INST0_SEG2', 'USB0_BASE__INST0_SEG3',
    'USB0_BASE__INST0_SEG4', 'USB0_BASE__INST1_SEG0',
    'USB0_BASE__INST1_SEG1', 'USB0_BASE__INST1_SEG2',
    'USB0_BASE__INST1_SEG3', 'USB0_BASE__INST1_SEG4',
    'USB0_BASE__INST2_SEG0', 'USB0_BASE__INST2_SEG1',
    'USB0_BASE__INST2_SEG2', 'USB0_BASE__INST2_SEG3',
    'USB0_BASE__INST2_SEG4', 'USB0_BASE__INST3_SEG0',
    'USB0_BASE__INST3_SEG1', 'USB0_BASE__INST3_SEG2',
    'USB0_BASE__INST3_SEG3', 'USB0_BASE__INST3_SEG4',
    'USB0_BASE__INST4_SEG0', 'USB0_BASE__INST4_SEG1',
    'USB0_BASE__INST4_SEG2', 'USB0_BASE__INST4_SEG3',
    'USB0_BASE__INST4_SEG4', 'USB0_BASE__INST5_SEG0',
    'USB0_BASE__INST5_SEG1', 'USB0_BASE__INST5_SEG2',
    'USB0_BASE__INST5_SEG3', 'USB0_BASE__INST5_SEG4',
    'USB0_BASE__INST6_SEG0', 'USB0_BASE__INST6_SEG1',
    'USB0_BASE__INST6_SEG2', 'USB0_BASE__INST6_SEG3',
    'USB0_BASE__INST6_SEG4', 'UVD0_BASE', 'UVD0_BASE__INST0_SEG0',
    'UVD0_BASE__INST0_SEG1', 'UVD0_BASE__INST0_SEG2',
    'UVD0_BASE__INST0_SEG3', 'UVD0_BASE__INST0_SEG4',
    'UVD0_BASE__INST1_SEG0', 'UVD0_BASE__INST1_SEG1',
    'UVD0_BASE__INST1_SEG2', 'UVD0_BASE__INST1_SEG3',
    'UVD0_BASE__INST1_SEG4', 'UVD0_BASE__INST2_SEG0',
    'UVD0_BASE__INST2_SEG1', 'UVD0_BASE__INST2_SEG2',
    'UVD0_BASE__INST2_SEG3', 'UVD0_BASE__INST2_SEG4',
    'UVD0_BASE__INST3_SEG0', 'UVD0_BASE__INST3_SEG1',
    'UVD0_BASE__INST3_SEG2', 'UVD0_BASE__INST3_SEG3',
    'UVD0_BASE__INST3_SEG4', 'UVD0_BASE__INST4_SEG0',
    'UVD0_BASE__INST4_SEG1', 'UVD0_BASE__INST4_SEG2',
    'UVD0_BASE__INST4_SEG3', 'UVD0_BASE__INST4_SEG4',
    'UVD0_BASE__INST5_SEG0', 'UVD0_BASE__INST5_SEG1',
    'UVD0_BASE__INST5_SEG2', 'UVD0_BASE__INST5_SEG3',
    'UVD0_BASE__INST5_SEG4', 'UVD0_BASE__INST6_SEG0',
    'UVD0_BASE__INST6_SEG1', 'UVD0_BASE__INST6_SEG2',
    'UVD0_BASE__INST6_SEG3', 'UVD0_BASE__INST6_SEG4',
    '__maybe_unused', '_navi14_ip_offset_HEADER', 'struct_IP_BASE',
    'struct_IP_BASE_INSTANCE']
