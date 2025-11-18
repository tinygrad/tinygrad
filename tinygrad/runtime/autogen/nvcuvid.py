# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/usr/local/cuda/include', '-D__CUVID_INTERNAL']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

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



_libraries = {}
_libraries['libnvcuvid.so'] = ctypes.CDLL('/lib/x86_64-linux-gnu/libnvcuvid.so')


CUvideodecoder = ctypes.POINTER(None)
class struct__CUcontextlock_st(Structure):
    pass

CUvideoctxlock = ctypes.POINTER(struct__CUcontextlock_st)

# values for enumeration 'cudaVideoCodec_enum'
cudaVideoCodec_enum__enumvalues = {
    0: 'cudaVideoCodec_MPEG1',
    1: 'cudaVideoCodec_MPEG2',
    2: 'cudaVideoCodec_MPEG4',
    3: 'cudaVideoCodec_VC1',
    4: 'cudaVideoCodec_H264',
    5: 'cudaVideoCodec_JPEG',
    6: 'cudaVideoCodec_H264_SVC',
    7: 'cudaVideoCodec_H264_MVC',
    8: 'cudaVideoCodec_HEVC',
    9: 'cudaVideoCodec_VP8',
    10: 'cudaVideoCodec_VP9',
    11: 'cudaVideoCodec_AV1',
    12: 'cudaVideoCodec_NumCodecs',
    1230591318: 'cudaVideoCodec_YUV420',
    1498820914: 'cudaVideoCodec_YV12',
    1314271538: 'cudaVideoCodec_NV12',
    1498765654: 'cudaVideoCodec_YUYV',
    1431918169: 'cudaVideoCodec_UYVY',
}
cudaVideoCodec_MPEG1 = 0
cudaVideoCodec_MPEG2 = 1
cudaVideoCodec_MPEG4 = 2
cudaVideoCodec_VC1 = 3
cudaVideoCodec_H264 = 4
cudaVideoCodec_JPEG = 5
cudaVideoCodec_H264_SVC = 6
cudaVideoCodec_H264_MVC = 7
cudaVideoCodec_HEVC = 8
cudaVideoCodec_VP8 = 9
cudaVideoCodec_VP9 = 10
cudaVideoCodec_AV1 = 11
cudaVideoCodec_NumCodecs = 12
cudaVideoCodec_YUV420 = 1230591318
cudaVideoCodec_YV12 = 1498820914
cudaVideoCodec_NV12 = 1314271538
cudaVideoCodec_YUYV = 1498765654
cudaVideoCodec_UYVY = 1431918169
cudaVideoCodec_enum = ctypes.c_uint32 # enum
cudaVideoCodec = cudaVideoCodec_enum
cudaVideoCodec__enumvalues = cudaVideoCodec_enum__enumvalues

# values for enumeration 'cudaVideoSurfaceFormat_enum'
cudaVideoSurfaceFormat_enum__enumvalues = {
    0: 'cudaVideoSurfaceFormat_NV12',
    1: 'cudaVideoSurfaceFormat_P016',
    2: 'cudaVideoSurfaceFormat_YUV444',
    3: 'cudaVideoSurfaceFormat_YUV444_16Bit',
    4: 'cudaVideoSurfaceFormat_NV16',
    5: 'cudaVideoSurfaceFormat_P216',
}
cudaVideoSurfaceFormat_NV12 = 0
cudaVideoSurfaceFormat_P016 = 1
cudaVideoSurfaceFormat_YUV444 = 2
cudaVideoSurfaceFormat_YUV444_16Bit = 3
cudaVideoSurfaceFormat_NV16 = 4
cudaVideoSurfaceFormat_P216 = 5
cudaVideoSurfaceFormat_enum = ctypes.c_uint32 # enum
cudaVideoSurfaceFormat = cudaVideoSurfaceFormat_enum
cudaVideoSurfaceFormat__enumvalues = cudaVideoSurfaceFormat_enum__enumvalues

# values for enumeration 'cudaVideoDeinterlaceMode_enum'
cudaVideoDeinterlaceMode_enum__enumvalues = {
    0: 'cudaVideoDeinterlaceMode_Weave',
    1: 'cudaVideoDeinterlaceMode_Bob',
    2: 'cudaVideoDeinterlaceMode_Adaptive',
}
cudaVideoDeinterlaceMode_Weave = 0
cudaVideoDeinterlaceMode_Bob = 1
cudaVideoDeinterlaceMode_Adaptive = 2
cudaVideoDeinterlaceMode_enum = ctypes.c_uint32 # enum
cudaVideoDeinterlaceMode = cudaVideoDeinterlaceMode_enum
cudaVideoDeinterlaceMode__enumvalues = cudaVideoDeinterlaceMode_enum__enumvalues

# values for enumeration 'cudaVideoChromaFormat_enum'
cudaVideoChromaFormat_enum__enumvalues = {
    0: 'cudaVideoChromaFormat_Monochrome',
    1: 'cudaVideoChromaFormat_420',
    2: 'cudaVideoChromaFormat_422',
    3: 'cudaVideoChromaFormat_444',
}
cudaVideoChromaFormat_Monochrome = 0
cudaVideoChromaFormat_420 = 1
cudaVideoChromaFormat_422 = 2
cudaVideoChromaFormat_444 = 3
cudaVideoChromaFormat_enum = ctypes.c_uint32 # enum
cudaVideoChromaFormat = cudaVideoChromaFormat_enum
cudaVideoChromaFormat__enumvalues = cudaVideoChromaFormat_enum__enumvalues

# values for enumeration 'cudaVideoCreateFlags_enum'
cudaVideoCreateFlags_enum__enumvalues = {
    0: 'cudaVideoCreate_Default',
    1: 'cudaVideoCreate_PreferCUDA',
    2: 'cudaVideoCreate_PreferDXVA',
    4: 'cudaVideoCreate_PreferCUVID',
}
cudaVideoCreate_Default = 0
cudaVideoCreate_PreferCUDA = 1
cudaVideoCreate_PreferDXVA = 2
cudaVideoCreate_PreferCUVID = 4
cudaVideoCreateFlags_enum = ctypes.c_uint32 # enum
cudaVideoCreateFlags = cudaVideoCreateFlags_enum
cudaVideoCreateFlags__enumvalues = cudaVideoCreateFlags_enum__enumvalues

# values for enumeration 'cuvidDecodeStatus_enum'
cuvidDecodeStatus_enum__enumvalues = {
    0: 'cuvidDecodeStatus_Invalid',
    1: 'cuvidDecodeStatus_InProgress',
    2: 'cuvidDecodeStatus_Success',
    8: 'cuvidDecodeStatus_Error',
    9: 'cuvidDecodeStatus_Error_Concealed',
}
cuvidDecodeStatus_Invalid = 0
cuvidDecodeStatus_InProgress = 1
cuvidDecodeStatus_Success = 2
cuvidDecodeStatus_Error = 8
cuvidDecodeStatus_Error_Concealed = 9
cuvidDecodeStatus_enum = ctypes.c_uint32 # enum
cuvidDecodeStatus = cuvidDecodeStatus_enum
cuvidDecodeStatus__enumvalues = cuvidDecodeStatus_enum__enumvalues
class struct__CUVIDDECODECAPS(Structure):
    pass

struct__CUVIDDECODECAPS._pack_ = 1 # source:False
struct__CUVIDDECODECAPS._fields_ = [
    ('eCodecType', cudaVideoCodec),
    ('eChromaFormat', cudaVideoChromaFormat),
    ('nBitDepthMinus8', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32 * 3),
    ('bIsSupported', ctypes.c_ubyte),
    ('nNumNVDECs', ctypes.c_ubyte),
    ('nOutputFormatMask', ctypes.c_uint16),
    ('nMaxWidth', ctypes.c_uint32),
    ('nMaxHeight', ctypes.c_uint32),
    ('nMaxMBCount', ctypes.c_uint32),
    ('nMinWidth', ctypes.c_uint16),
    ('nMinHeight', ctypes.c_uint16),
    ('bIsHistogramSupported', ctypes.c_ubyte),
    ('nCounterBitDepth', ctypes.c_ubyte),
    ('nMaxHistogramBins', ctypes.c_uint16),
    ('reserved3', ctypes.c_uint32 * 10),
]

CUVIDDECODECAPS = struct__CUVIDDECODECAPS
class struct__CUVIDDECODECREATEINFO(Structure):
    pass

class struct__CUVIDDECODECREATEINFO_display_area(Structure):
    pass

struct__CUVIDDECODECREATEINFO_display_area._pack_ = 1 # source:False
struct__CUVIDDECODECREATEINFO_display_area._fields_ = [
    ('left', ctypes.c_int16),
    ('top', ctypes.c_int16),
    ('right', ctypes.c_int16),
    ('bottom', ctypes.c_int16),
]

class struct__CUVIDDECODECREATEINFO_target_rect(Structure):
    pass

struct__CUVIDDECODECREATEINFO_target_rect._pack_ = 1 # source:False
struct__CUVIDDECODECREATEINFO_target_rect._fields_ = [
    ('left', ctypes.c_int16),
    ('top', ctypes.c_int16),
    ('right', ctypes.c_int16),
    ('bottom', ctypes.c_int16),
]

struct__CUVIDDECODECREATEINFO._pack_ = 1 # source:False
struct__CUVIDDECODECREATEINFO._fields_ = [
    ('ulWidth', ctypes.c_uint64),
    ('ulHeight', ctypes.c_uint64),
    ('ulNumDecodeSurfaces', ctypes.c_uint64),
    ('CodecType', cudaVideoCodec),
    ('ChromaFormat', cudaVideoChromaFormat),
    ('ulCreationFlags', ctypes.c_uint64),
    ('bitDepthMinus8', ctypes.c_uint64),
    ('ulIntraDecodeOnly', ctypes.c_uint64),
    ('ulMaxWidth', ctypes.c_uint64),
    ('ulMaxHeight', ctypes.c_uint64),
    ('Reserved1', ctypes.c_uint64),
    ('display_area', struct__CUVIDDECODECREATEINFO_display_area),
    ('OutputFormat', cudaVideoSurfaceFormat),
    ('DeinterlaceMode', cudaVideoDeinterlaceMode),
    ('ulTargetWidth', ctypes.c_uint64),
    ('ulTargetHeight', ctypes.c_uint64),
    ('ulNumOutputSurfaces', ctypes.c_uint64),
    ('vidLock', ctypes.POINTER(struct__CUcontextlock_st)),
    ('target_rect', struct__CUVIDDECODECREATEINFO_target_rect),
    ('enableHistogram', ctypes.c_uint64),
    ('Reserved2', ctypes.c_uint64 * 4),
]

CUVIDDECODECREATEINFO = struct__CUVIDDECODECREATEINFO
class struct__CUVIDH264DPBENTRY(Structure):
    pass

struct__CUVIDH264DPBENTRY._pack_ = 1 # source:False
struct__CUVIDH264DPBENTRY._fields_ = [
    ('PicIdx', ctypes.c_int32),
    ('FrameIdx', ctypes.c_int32),
    ('is_long_term', ctypes.c_int32),
    ('not_existing', ctypes.c_int32),
    ('used_for_reference', ctypes.c_int32),
    ('FieldOrderCnt', ctypes.c_int32 * 2),
]

CUVIDH264DPBENTRY = struct__CUVIDH264DPBENTRY
class struct__CUVIDH264MVCEXT(Structure):
    pass

struct__CUVIDH264MVCEXT._pack_ = 1 # source:False
struct__CUVIDH264MVCEXT._fields_ = [
    ('num_views_minus1', ctypes.c_int32),
    ('view_id', ctypes.c_int32),
    ('inter_view_flag', ctypes.c_ubyte),
    ('num_inter_view_refs_l0', ctypes.c_ubyte),
    ('num_inter_view_refs_l1', ctypes.c_ubyte),
    ('MVCReserved8Bits', ctypes.c_ubyte),
    ('InterViewRefsL0', ctypes.c_int32 * 16),
    ('InterViewRefsL1', ctypes.c_int32 * 16),
]

CUVIDH264MVCEXT = struct__CUVIDH264MVCEXT
class struct__CUVIDH264SVCEXT(Structure):
    pass

class struct__CUVIDPICPARAMS(Structure):
    pass

struct__CUVIDH264SVCEXT._pack_ = 1 # source:False
struct__CUVIDH264SVCEXT._fields_ = [
    ('profile_idc', ctypes.c_ubyte),
    ('level_idc', ctypes.c_ubyte),
    ('DQId', ctypes.c_ubyte),
    ('DQIdMax', ctypes.c_ubyte),
    ('disable_inter_layer_deblocking_filter_idc', ctypes.c_ubyte),
    ('ref_layer_chroma_phase_y_plus1', ctypes.c_ubyte),
    ('inter_layer_slice_alpha_c0_offset_div2', ctypes.c_byte),
    ('inter_layer_slice_beta_offset_div2', ctypes.c_byte),
    ('DPBEntryValidFlag', ctypes.c_uint16),
    ('inter_layer_deblocking_filter_control_present_flag', ctypes.c_ubyte),
    ('extended_spatial_scalability_idc', ctypes.c_ubyte),
    ('adaptive_tcoeff_level_prediction_flag', ctypes.c_ubyte),
    ('slice_header_restriction_flag', ctypes.c_ubyte),
    ('chroma_phase_x_plus1_flag', ctypes.c_ubyte),
    ('chroma_phase_y_plus1', ctypes.c_ubyte),
    ('tcoeff_level_prediction_flag', ctypes.c_ubyte),
    ('constrained_intra_resampling_flag', ctypes.c_ubyte),
    ('ref_layer_chroma_phase_x_plus1_flag', ctypes.c_ubyte),
    ('store_ref_base_pic_flag', ctypes.c_ubyte),
    ('Reserved8BitsA', ctypes.c_ubyte),
    ('Reserved8BitsB', ctypes.c_ubyte),
    ('scaled_ref_layer_left_offset', ctypes.c_int16),
    ('scaled_ref_layer_top_offset', ctypes.c_int16),
    ('scaled_ref_layer_right_offset', ctypes.c_int16),
    ('scaled_ref_layer_bottom_offset', ctypes.c_int16),
    ('Reserved16Bits', ctypes.c_uint16),
    ('pNextLayer', ctypes.POINTER(struct__CUVIDPICPARAMS)),
    ('bRefBaseLayer', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class union__CUVIDPICPARAMS_CodecSpecific(Union):
    pass

class struct__CUVIDMPEG2PICPARAMS(Structure):
    pass

struct__CUVIDMPEG2PICPARAMS._pack_ = 1 # source:False
struct__CUVIDMPEG2PICPARAMS._fields_ = [
    ('ForwardRefIdx', ctypes.c_int32),
    ('BackwardRefIdx', ctypes.c_int32),
    ('picture_coding_type', ctypes.c_int32),
    ('full_pel_forward_vector', ctypes.c_int32),
    ('full_pel_backward_vector', ctypes.c_int32),
    ('f_code', ctypes.c_int32 * 2 * 2),
    ('intra_dc_precision', ctypes.c_int32),
    ('frame_pred_frame_dct', ctypes.c_int32),
    ('concealment_motion_vectors', ctypes.c_int32),
    ('q_scale_type', ctypes.c_int32),
    ('intra_vlc_format', ctypes.c_int32),
    ('alternate_scan', ctypes.c_int32),
    ('top_field_first', ctypes.c_int32),
    ('QuantMatrixIntra', ctypes.c_ubyte * 64),
    ('QuantMatrixInter', ctypes.c_ubyte * 64),
]

class struct__CUVIDH264PICPARAMS(Structure):
    pass

class union__CUVIDH264PICPARAMS_fmo(Union):
    pass

union__CUVIDH264PICPARAMS_fmo._pack_ = 1 # source:False
union__CUVIDH264PICPARAMS_fmo._fields_ = [
    ('slice_group_map_addr', ctypes.c_uint64),
    ('pMb2SliceGroupMap', ctypes.POINTER(ctypes.c_ubyte)),
]

class union__CUVIDH264PICPARAMS_1(Union):
    pass

union__CUVIDH264PICPARAMS_1._pack_ = 1 # source:False
union__CUVIDH264PICPARAMS_1._fields_ = [
    ('mvcext', CUVIDH264MVCEXT),
    ('svcext', struct__CUVIDH264SVCEXT),
    ('PADDING_0', ctypes.c_ubyte * 96),
]

struct__CUVIDH264PICPARAMS._pack_ = 1 # source:False
struct__CUVIDH264PICPARAMS._anonymous_ = ('_0',)
struct__CUVIDH264PICPARAMS._fields_ = [
    ('log2_max_frame_num_minus4', ctypes.c_int32),
    ('pic_order_cnt_type', ctypes.c_int32),
    ('log2_max_pic_order_cnt_lsb_minus4', ctypes.c_int32),
    ('delta_pic_order_always_zero_flag', ctypes.c_int32),
    ('frame_mbs_only_flag', ctypes.c_int32),
    ('direct_8x8_inference_flag', ctypes.c_int32),
    ('num_ref_frames', ctypes.c_int32),
    ('residual_colour_transform_flag', ctypes.c_ubyte),
    ('bit_depth_luma_minus8', ctypes.c_ubyte),
    ('bit_depth_chroma_minus8', ctypes.c_ubyte),
    ('qpprime_y_zero_transform_bypass_flag', ctypes.c_ubyte),
    ('entropy_coding_mode_flag', ctypes.c_int32),
    ('pic_order_present_flag', ctypes.c_int32),
    ('num_ref_idx_l0_active_minus1', ctypes.c_int32),
    ('num_ref_idx_l1_active_minus1', ctypes.c_int32),
    ('weighted_pred_flag', ctypes.c_int32),
    ('weighted_bipred_idc', ctypes.c_int32),
    ('pic_init_qp_minus26', ctypes.c_int32),
    ('deblocking_filter_control_present_flag', ctypes.c_int32),
    ('redundant_pic_cnt_present_flag', ctypes.c_int32),
    ('transform_8x8_mode_flag', ctypes.c_int32),
    ('MbaffFrameFlag', ctypes.c_int32),
    ('constrained_intra_pred_flag', ctypes.c_int32),
    ('chroma_qp_index_offset', ctypes.c_int32),
    ('second_chroma_qp_index_offset', ctypes.c_int32),
    ('ref_pic_flag', ctypes.c_int32),
    ('frame_num', ctypes.c_int32),
    ('CurrFieldOrderCnt', ctypes.c_int32 * 2),
    ('dpb', struct__CUVIDH264DPBENTRY * 16),
    ('WeightScale4x4', ctypes.c_ubyte * 16 * 6),
    ('WeightScale8x8', ctypes.c_ubyte * 64 * 2),
    ('fmo_aso_enable', ctypes.c_ubyte),
    ('num_slice_groups_minus1', ctypes.c_ubyte),
    ('slice_group_map_type', ctypes.c_ubyte),
    ('pic_init_qs_minus26', ctypes.c_byte),
    ('slice_group_change_rate_minus1', ctypes.c_uint32),
    ('fmo', union__CUVIDH264PICPARAMS_fmo),
    ('mb_adaptive_frame_field_flag', ctypes.c_uint32, 2),
    ('Reserved1', ctypes.c_uint32, 30),
    ('Reserved', ctypes.c_uint32 * 11),
    ('_0', union__CUVIDH264PICPARAMS_1),
]

class struct__CUVIDVC1PICPARAMS(Structure):
    pass

struct__CUVIDVC1PICPARAMS._pack_ = 1 # source:False
struct__CUVIDVC1PICPARAMS._fields_ = [
    ('ForwardRefIdx', ctypes.c_int32),
    ('BackwardRefIdx', ctypes.c_int32),
    ('FrameWidth', ctypes.c_int32),
    ('FrameHeight', ctypes.c_int32),
    ('intra_pic_flag', ctypes.c_int32),
    ('ref_pic_flag', ctypes.c_int32),
    ('progressive_fcm', ctypes.c_int32),
    ('profile', ctypes.c_int32),
    ('postprocflag', ctypes.c_int32),
    ('pulldown', ctypes.c_int32),
    ('interlace', ctypes.c_int32),
    ('tfcntrflag', ctypes.c_int32),
    ('finterpflag', ctypes.c_int32),
    ('psf', ctypes.c_int32),
    ('multires', ctypes.c_int32),
    ('syncmarker', ctypes.c_int32),
    ('rangered', ctypes.c_int32),
    ('maxbframes', ctypes.c_int32),
    ('panscan_flag', ctypes.c_int32),
    ('refdist_flag', ctypes.c_int32),
    ('extended_mv', ctypes.c_int32),
    ('dquant', ctypes.c_int32),
    ('vstransform', ctypes.c_int32),
    ('loopfilter', ctypes.c_int32),
    ('fastuvmc', ctypes.c_int32),
    ('overlap', ctypes.c_int32),
    ('quantizer', ctypes.c_int32),
    ('extended_dmv', ctypes.c_int32),
    ('range_mapy_flag', ctypes.c_int32),
    ('range_mapy', ctypes.c_int32),
    ('range_mapuv_flag', ctypes.c_int32),
    ('range_mapuv', ctypes.c_int32),
    ('rangeredfrm', ctypes.c_int32),
]

class struct__CUVIDMPEG4PICPARAMS(Structure):
    pass

struct__CUVIDMPEG4PICPARAMS._pack_ = 1 # source:False
struct__CUVIDMPEG4PICPARAMS._fields_ = [
    ('ForwardRefIdx', ctypes.c_int32),
    ('BackwardRefIdx', ctypes.c_int32),
    ('video_object_layer_width', ctypes.c_int32),
    ('video_object_layer_height', ctypes.c_int32),
    ('vop_time_increment_bitcount', ctypes.c_int32),
    ('top_field_first', ctypes.c_int32),
    ('resync_marker_disable', ctypes.c_int32),
    ('quant_type', ctypes.c_int32),
    ('quarter_sample', ctypes.c_int32),
    ('short_video_header', ctypes.c_int32),
    ('divx_flags', ctypes.c_int32),
    ('vop_coding_type', ctypes.c_int32),
    ('vop_coded', ctypes.c_int32),
    ('vop_rounding_type', ctypes.c_int32),
    ('alternate_vertical_scan_flag', ctypes.c_int32),
    ('interlaced', ctypes.c_int32),
    ('vop_fcode_forward', ctypes.c_int32),
    ('vop_fcode_backward', ctypes.c_int32),
    ('trd', ctypes.c_int32 * 2),
    ('trb', ctypes.c_int32 * 2),
    ('QuantMatrixIntra', ctypes.c_ubyte * 64),
    ('QuantMatrixInter', ctypes.c_ubyte * 64),
    ('gmc_enabled', ctypes.c_int32),
]

class struct__CUVIDJPEGPICPARAMS(Structure):
    pass

struct__CUVIDJPEGPICPARAMS._pack_ = 1 # source:False
struct__CUVIDJPEGPICPARAMS._fields_ = [
    ('numComponents', ctypes.c_ubyte),
    ('bitDepth', ctypes.c_ubyte),
    ('quantizationTableSelector', ctypes.c_ubyte * 4),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('scanOffset', ctypes.c_uint32 * 4),
    ('scanSize', ctypes.c_uint32 * 4),
    ('restartInterval', ctypes.c_uint16),
    ('componentIdentifier', ctypes.c_ubyte * 4),
    ('hasQMatrix', ctypes.c_ubyte),
    ('hasHuffman', ctypes.c_ubyte),
    ('quantvals', ctypes.c_uint16 * 64 * 4),
    ('bits_ac', ctypes.c_ubyte * 16 * 4),
    ('table_ac', ctypes.c_ubyte * 256 * 4),
    ('bits_dc', ctypes.c_ubyte * 16 * 4),
    ('table_dc', ctypes.c_ubyte * 256 * 4),
]

class struct__CUVIDHEVCPICPARAMS(Structure):
    pass

struct__CUVIDHEVCPICPARAMS._pack_ = 1 # source:False
struct__CUVIDHEVCPICPARAMS._fields_ = [
    ('pic_width_in_luma_samples', ctypes.c_int32),
    ('pic_height_in_luma_samples', ctypes.c_int32),
    ('log2_min_luma_coding_block_size_minus3', ctypes.c_ubyte),
    ('log2_diff_max_min_luma_coding_block_size', ctypes.c_ubyte),
    ('log2_min_transform_block_size_minus2', ctypes.c_ubyte),
    ('log2_diff_max_min_transform_block_size', ctypes.c_ubyte),
    ('pcm_enabled_flag', ctypes.c_ubyte),
    ('log2_min_pcm_luma_coding_block_size_minus3', ctypes.c_ubyte),
    ('log2_diff_max_min_pcm_luma_coding_block_size', ctypes.c_ubyte),
    ('pcm_sample_bit_depth_luma_minus1', ctypes.c_ubyte),
    ('pcm_sample_bit_depth_chroma_minus1', ctypes.c_ubyte),
    ('pcm_loop_filter_disabled_flag', ctypes.c_ubyte),
    ('strong_intra_smoothing_enabled_flag', ctypes.c_ubyte),
    ('max_transform_hierarchy_depth_intra', ctypes.c_ubyte),
    ('max_transform_hierarchy_depth_inter', ctypes.c_ubyte),
    ('amp_enabled_flag', ctypes.c_ubyte),
    ('separate_colour_plane_flag', ctypes.c_ubyte),
    ('log2_max_pic_order_cnt_lsb_minus4', ctypes.c_ubyte),
    ('num_short_term_ref_pic_sets', ctypes.c_ubyte),
    ('long_term_ref_pics_present_flag', ctypes.c_ubyte),
    ('num_long_term_ref_pics_sps', ctypes.c_ubyte),
    ('sps_temporal_mvp_enabled_flag', ctypes.c_ubyte),
    ('sample_adaptive_offset_enabled_flag', ctypes.c_ubyte),
    ('scaling_list_enable_flag', ctypes.c_ubyte),
    ('IrapPicFlag', ctypes.c_ubyte),
    ('IdrPicFlag', ctypes.c_ubyte),
    ('bit_depth_luma_minus8', ctypes.c_ubyte),
    ('bit_depth_chroma_minus8', ctypes.c_ubyte),
    ('log2_max_transform_skip_block_size_minus2', ctypes.c_ubyte),
    ('log2_sao_offset_scale_luma', ctypes.c_ubyte),
    ('log2_sao_offset_scale_chroma', ctypes.c_ubyte),
    ('high_precision_offsets_enabled_flag', ctypes.c_ubyte),
    ('reserved1', ctypes.c_ubyte * 10),
    ('dependent_slice_segments_enabled_flag', ctypes.c_ubyte),
    ('slice_segment_header_extension_present_flag', ctypes.c_ubyte),
    ('sign_data_hiding_enabled_flag', ctypes.c_ubyte),
    ('cu_qp_delta_enabled_flag', ctypes.c_ubyte),
    ('diff_cu_qp_delta_depth', ctypes.c_ubyte),
    ('init_qp_minus26', ctypes.c_byte),
    ('pps_cb_qp_offset', ctypes.c_byte),
    ('pps_cr_qp_offset', ctypes.c_byte),
    ('constrained_intra_pred_flag', ctypes.c_ubyte),
    ('weighted_pred_flag', ctypes.c_ubyte),
    ('weighted_bipred_flag', ctypes.c_ubyte),
    ('transform_skip_enabled_flag', ctypes.c_ubyte),
    ('transquant_bypass_enabled_flag', ctypes.c_ubyte),
    ('entropy_coding_sync_enabled_flag', ctypes.c_ubyte),
    ('log2_parallel_merge_level_minus2', ctypes.c_ubyte),
    ('num_extra_slice_header_bits', ctypes.c_ubyte),
    ('loop_filter_across_tiles_enabled_flag', ctypes.c_ubyte),
    ('loop_filter_across_slices_enabled_flag', ctypes.c_ubyte),
    ('output_flag_present_flag', ctypes.c_ubyte),
    ('num_ref_idx_l0_default_active_minus1', ctypes.c_ubyte),
    ('num_ref_idx_l1_default_active_minus1', ctypes.c_ubyte),
    ('lists_modification_present_flag', ctypes.c_ubyte),
    ('cabac_init_present_flag', ctypes.c_ubyte),
    ('pps_slice_chroma_qp_offsets_present_flag', ctypes.c_ubyte),
    ('deblocking_filter_override_enabled_flag', ctypes.c_ubyte),
    ('pps_deblocking_filter_disabled_flag', ctypes.c_ubyte),
    ('pps_beta_offset_div2', ctypes.c_byte),
    ('pps_tc_offset_div2', ctypes.c_byte),
    ('tiles_enabled_flag', ctypes.c_ubyte),
    ('uniform_spacing_flag', ctypes.c_ubyte),
    ('num_tile_columns_minus1', ctypes.c_ubyte),
    ('num_tile_rows_minus1', ctypes.c_ubyte),
    ('column_width_minus1', ctypes.c_uint16 * 21),
    ('row_height_minus1', ctypes.c_uint16 * 21),
    ('sps_range_extension_flag', ctypes.c_ubyte),
    ('transform_skip_rotation_enabled_flag', ctypes.c_ubyte),
    ('transform_skip_context_enabled_flag', ctypes.c_ubyte),
    ('implicit_rdpcm_enabled_flag', ctypes.c_ubyte),
    ('explicit_rdpcm_enabled_flag', ctypes.c_ubyte),
    ('extended_precision_processing_flag', ctypes.c_ubyte),
    ('intra_smoothing_disabled_flag', ctypes.c_ubyte),
    ('persistent_rice_adaptation_enabled_flag', ctypes.c_ubyte),
    ('cabac_bypass_alignment_enabled_flag', ctypes.c_ubyte),
    ('pps_range_extension_flag', ctypes.c_ubyte),
    ('cross_component_prediction_enabled_flag', ctypes.c_ubyte),
    ('chroma_qp_offset_list_enabled_flag', ctypes.c_ubyte),
    ('diff_cu_chroma_qp_offset_depth', ctypes.c_ubyte),
    ('chroma_qp_offset_list_len_minus1', ctypes.c_ubyte),
    ('cb_qp_offset_list', ctypes.c_byte * 6),
    ('cr_qp_offset_list', ctypes.c_byte * 6),
    ('reserved2', ctypes.c_ubyte * 2),
    ('reserved3', ctypes.c_uint32 * 8),
    ('NumBitsForShortTermRPSInSlice', ctypes.c_int32),
    ('NumDeltaPocsOfRefRpsIdx', ctypes.c_int32),
    ('NumPocTotalCurr', ctypes.c_int32),
    ('NumPocStCurrBefore', ctypes.c_int32),
    ('NumPocStCurrAfter', ctypes.c_int32),
    ('NumPocLtCurr', ctypes.c_int32),
    ('CurrPicOrderCntVal', ctypes.c_int32),
    ('RefPicIdx', ctypes.c_int32 * 16),
    ('PicOrderCntVal', ctypes.c_int32 * 16),
    ('IsLongTerm', ctypes.c_ubyte * 16),
    ('RefPicSetStCurrBefore', ctypes.c_ubyte * 8),
    ('RefPicSetStCurrAfter', ctypes.c_ubyte * 8),
    ('RefPicSetLtCurr', ctypes.c_ubyte * 8),
    ('RefPicSetInterLayer0', ctypes.c_ubyte * 8),
    ('RefPicSetInterLayer1', ctypes.c_ubyte * 8),
    ('reserved4', ctypes.c_uint32 * 12),
    ('ScalingList4x4', ctypes.c_ubyte * 16 * 6),
    ('ScalingList8x8', ctypes.c_ubyte * 64 * 6),
    ('ScalingList16x16', ctypes.c_ubyte * 64 * 6),
    ('ScalingList32x32', ctypes.c_ubyte * 64 * 2),
    ('ScalingListDCCoeff16x16', ctypes.c_ubyte * 6),
    ('ScalingListDCCoeff32x32', ctypes.c_ubyte * 2),
]

class struct__CUVIDVP8PICPARAMS(Structure):
    pass

class union__CUVIDVP8PICPARAMS_0(Union):
    pass

class struct__CUVIDVP8PICPARAMS_0_vp8_frame_tag(Structure):
    pass

struct__CUVIDVP8PICPARAMS_0_vp8_frame_tag._pack_ = 1 # source:False
struct__CUVIDVP8PICPARAMS_0_vp8_frame_tag._fields_ = [
    ('frame_type', ctypes.c_ubyte, 1),
    ('version', ctypes.c_ubyte, 3),
    ('show_frame', ctypes.c_ubyte, 1),
    ('update_mb_segmentation_data', ctypes.c_ubyte, 1),
    ('Reserved2Bits', ctypes.c_ubyte, 2),
]

union__CUVIDVP8PICPARAMS_0._pack_ = 1 # source:False
union__CUVIDVP8PICPARAMS_0._fields_ = [
    ('vp8_frame_tag', struct__CUVIDVP8PICPARAMS_0_vp8_frame_tag),
    ('wFrameTagFlags', ctypes.c_ubyte),
]

struct__CUVIDVP8PICPARAMS._pack_ = 1 # source:False
struct__CUVIDVP8PICPARAMS._anonymous_ = ('_0',)
struct__CUVIDVP8PICPARAMS._fields_ = [
    ('width', ctypes.c_int32),
    ('height', ctypes.c_int32),
    ('first_partition_size', ctypes.c_uint32),
    ('LastRefIdx', ctypes.c_ubyte),
    ('GoldenRefIdx', ctypes.c_ubyte),
    ('AltRefIdx', ctypes.c_ubyte),
    ('_0', union__CUVIDVP8PICPARAMS_0),
    ('Reserved1', ctypes.c_ubyte * 4),
    ('Reserved2', ctypes.c_uint32 * 3),
]

class struct__CUVIDVP9PICPARAMS(Structure):
    pass

struct__CUVIDVP9PICPARAMS._pack_ = 1 # source:False
struct__CUVIDVP9PICPARAMS._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('LastRefIdx', ctypes.c_ubyte),
    ('GoldenRefIdx', ctypes.c_ubyte),
    ('AltRefIdx', ctypes.c_ubyte),
    ('colorSpace', ctypes.c_ubyte),
    ('profile', ctypes.c_uint16, 3),
    ('frameContextIdx', ctypes.c_uint16, 2),
    ('frameType', ctypes.c_uint16, 1),
    ('showFrame', ctypes.c_uint16, 1),
    ('errorResilient', ctypes.c_uint16, 1),
    ('frameParallelDecoding', ctypes.c_uint16, 1),
    ('subSamplingX', ctypes.c_uint16, 1),
    ('subSamplingY', ctypes.c_uint16, 1),
    ('intraOnly', ctypes.c_uint16, 1),
    ('allow_high_precision_mv', ctypes.c_uint16, 1),
    ('refreshEntropyProbs', ctypes.c_uint16, 1),
    ('reserved2Bits', ctypes.c_uint16, 2),
    ('reserved16Bits', ctypes.c_uint16),
    ('refFrameSignBias', ctypes.c_ubyte * 4),
    ('bitDepthMinus8Luma', ctypes.c_ubyte),
    ('bitDepthMinus8Chroma', ctypes.c_ubyte),
    ('loopFilterLevel', ctypes.c_ubyte),
    ('loopFilterSharpness', ctypes.c_ubyte),
    ('modeRefLfEnabled', ctypes.c_ubyte),
    ('log2_tile_columns', ctypes.c_ubyte),
    ('log2_tile_rows', ctypes.c_ubyte),
    ('segmentEnabled', ctypes.c_ubyte, 1),
    ('segmentMapUpdate', ctypes.c_ubyte, 1),
    ('segmentMapTemporalUpdate', ctypes.c_ubyte, 1),
    ('segmentFeatureMode', ctypes.c_ubyte, 1),
    ('reserved4Bits', ctypes.c_ubyte, 4),
    ('segmentFeatureEnable', ctypes.c_ubyte * 4 * 8),
    ('segmentFeatureData', ctypes.c_int16 * 4 * 8),
    ('mb_segment_tree_probs', ctypes.c_ubyte * 7),
    ('segment_pred_probs', ctypes.c_ubyte * 3),
    ('reservedSegment16Bits', ctypes.c_ubyte * 2),
    ('qpYAc', ctypes.c_int32),
    ('qpYDc', ctypes.c_int32),
    ('qpChDc', ctypes.c_int32),
    ('qpChAc', ctypes.c_int32),
    ('activeRefIdx', ctypes.c_uint32 * 3),
    ('resetFrameContext', ctypes.c_uint32),
    ('mcomp_filter_type', ctypes.c_uint32),
    ('mbRefLfDelta', ctypes.c_uint32 * 4),
    ('mbModeLfDelta', ctypes.c_uint32 * 2),
    ('frameTagSize', ctypes.c_uint32),
    ('offsetToDctParts', ctypes.c_uint32),
    ('reserved128Bits', ctypes.c_uint32 * 4),
]

class struct__CUVIDAV1PICPARAMS(Structure):
    pass

class struct__CUVIDAV1PICPARAMS_0(Structure):
    pass

struct__CUVIDAV1PICPARAMS_0._pack_ = 1 # source:False
struct__CUVIDAV1PICPARAMS_0._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('index', ctypes.c_ubyte),
    ('reserved24Bits', ctypes.c_ubyte * 3),
]

class struct__CUVIDAV1PICPARAMS_1(Structure):
    pass

struct__CUVIDAV1PICPARAMS_1._pack_ = 1 # source:False
struct__CUVIDAV1PICPARAMS_1._fields_ = [
    ('invalid', ctypes.c_ubyte, 1),
    ('wmtype', ctypes.c_ubyte, 2),
    ('reserved5Bits', ctypes.c_ubyte, 5),
    ('reserved24Bits', ctypes.c_char * 3),
    ('wmmat', ctypes.c_int32 * 6),
]

struct__CUVIDAV1PICPARAMS._pack_ = 1 # source:False
struct__CUVIDAV1PICPARAMS._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('frame_offset', ctypes.c_uint32),
    ('decodePicIdx', ctypes.c_int32),
    ('profile', ctypes.c_uint64, 3),
    ('use_128x128_superblock', ctypes.c_uint64, 1),
    ('subsampling_x', ctypes.c_uint64, 1),
    ('subsampling_y', ctypes.c_uint64, 1),
    ('mono_chrome', ctypes.c_uint64, 1),
    ('bit_depth_minus8', ctypes.c_uint64, 4),
    ('enable_filter_intra', ctypes.c_uint64, 1),
    ('enable_intra_edge_filter', ctypes.c_uint64, 1),
    ('enable_interintra_compound', ctypes.c_uint64, 1),
    ('enable_masked_compound', ctypes.c_uint64, 1),
    ('enable_dual_filter', ctypes.c_uint64, 1),
    ('enable_order_hint', ctypes.c_uint64, 1),
    ('order_hint_bits_minus1', ctypes.c_uint64, 3),
    ('enable_jnt_comp', ctypes.c_uint64, 1),
    ('enable_superres', ctypes.c_uint64, 1),
    ('enable_cdef', ctypes.c_uint64, 1),
    ('enable_restoration', ctypes.c_uint64, 1),
    ('enable_fgs', ctypes.c_uint64, 1),
    ('reserved0_7bits', ctypes.c_uint64, 7),
    ('frame_type', ctypes.c_uint64, 2),
    ('show_frame', ctypes.c_uint64, 1),
    ('disable_cdf_update', ctypes.c_uint64, 1),
    ('allow_screen_content_tools', ctypes.c_uint64, 1),
    ('force_integer_mv', ctypes.c_uint64, 1),
    ('coded_denom', ctypes.c_uint64, 3),
    ('allow_intrabc', ctypes.c_uint64, 1),
    ('allow_high_precision_mv', ctypes.c_uint64, 1),
    ('interp_filter', ctypes.c_uint64, 3),
    ('switchable_motion_mode', ctypes.c_uint64, 1),
    ('use_ref_frame_mvs', ctypes.c_uint64, 1),
    ('disable_frame_end_update_cdf', ctypes.c_uint64, 1),
    ('delta_q_present', ctypes.c_uint64, 1),
    ('delta_q_res', ctypes.c_uint64, 2),
    ('using_qmatrix', ctypes.c_uint64, 1),
    ('coded_lossless', ctypes.c_uint64, 1),
    ('use_superres', ctypes.c_uint64, 1),
    ('tx_mode', ctypes.c_uint64, 2),
    ('reference_mode', ctypes.c_uint64, 1),
    ('allow_warped_motion', ctypes.c_uint64, 1),
    ('reduced_tx_set', ctypes.c_uint64, 1),
    ('skip_mode', ctypes.c_uint64, 1),
    ('reserved1_3bits', ctypes.c_uint64, 3),
    ('num_tile_cols', ctypes.c_uint64, 8),
    ('num_tile_rows', ctypes.c_uint64, 8),
    ('context_update_tile_id', ctypes.c_uint64, 16),
    ('tile_widths', ctypes.c_uint16 * 64),
    ('tile_heights', ctypes.c_uint16 * 64),
    ('cdef_damping_minus_3', ctypes.c_ubyte, 2),
    ('cdef_bits', ctypes.c_ubyte, 2),
    ('reserved2_4bits', ctypes.c_ubyte, 4),
    ('cdef_y_strength', ctypes.c_ubyte * 8),
    ('cdef_uv_strength', ctypes.c_ubyte * 8),
    ('SkipModeFrame0', ctypes.c_ubyte, 4),
    ('SkipModeFrame1', ctypes.c_ubyte, 4),
    ('base_qindex', ctypes.c_ubyte, 8),
    ('qp_y_dc_delta_q', ctypes.c_char),
    ('qp_u_dc_delta_q', ctypes.c_char),
    ('qp_v_dc_delta_q', ctypes.c_char),
    ('qp_u_ac_delta_q', ctypes.c_char),
    ('qp_v_ac_delta_q', ctypes.c_char),
    ('qm_y', ctypes.c_ubyte),
    ('qm_u', ctypes.c_ubyte),
    ('qm_v', ctypes.c_ubyte),
    ('segmentation_enabled', ctypes.c_ubyte, 1),
    ('segmentation_update_map', ctypes.c_ubyte, 1),
    ('segmentation_update_data', ctypes.c_ubyte, 1),
    ('segmentation_temporal_update', ctypes.c_ubyte, 1),
    ('reserved3_4bits', ctypes.c_ubyte, 4),
    ('segmentation_feature_data', ctypes.c_int16 * 8 * 8),
    ('segmentation_feature_mask', ctypes.c_ubyte * 8),
    ('loop_filter_level', ctypes.c_ubyte * 2),
    ('loop_filter_level_u', ctypes.c_ubyte),
    ('loop_filter_level_v', ctypes.c_ubyte),
    ('loop_filter_sharpness', ctypes.c_ubyte),
    ('loop_filter_ref_deltas', ctypes.c_char * 8),
    ('loop_filter_mode_deltas', ctypes.c_char * 2),
    ('loop_filter_delta_enabled', ctypes.c_ubyte, 1),
    ('loop_filter_delta_update', ctypes.c_ubyte, 1),
    ('delta_lf_present', ctypes.c_ubyte, 1),
    ('delta_lf_res', ctypes.c_ubyte, 2),
    ('delta_lf_multi', ctypes.c_ubyte, 1),
    ('reserved4_2bits', ctypes.c_ubyte, 2),
    ('lr_unit_size', ctypes.c_ubyte * 3),
    ('lr_type', ctypes.c_ubyte * 3),
    ('primary_ref_frame', ctypes.c_ubyte),
    ('ref_frame_map', ctypes.c_ubyte * 8),
    ('temporal_layer_id', ctypes.c_ubyte, 4),
    ('spatial_layer_id', ctypes.c_ubyte, 4),
    ('reserved5_32bits', ctypes.c_ubyte * 4),
    ('ref_frame', struct__CUVIDAV1PICPARAMS_0 * 7),
    ('global_motion', struct__CUVIDAV1PICPARAMS_1 * 7),
    ('apply_grain', ctypes.c_uint16, 1),
    ('overlap_flag', ctypes.c_uint16, 1),
    ('scaling_shift_minus8', ctypes.c_uint16, 2),
    ('chroma_scaling_from_luma', ctypes.c_uint16, 1),
    ('ar_coeff_lag', ctypes.c_uint16, 2),
    ('ar_coeff_shift_minus6', ctypes.c_uint16, 2),
    ('grain_scale_shift', ctypes.c_uint16, 2),
    ('clip_to_restricted_range', ctypes.c_uint16, 1),
    ('reserved6_4bits', ctypes.c_uint16, 4),
    ('num_y_points', ctypes.c_uint16, 8),
    ('scaling_points_y', ctypes.c_ubyte * 2 * 14),
    ('num_cb_points', ctypes.c_ubyte),
    ('scaling_points_cb', ctypes.c_ubyte * 2 * 10),
    ('num_cr_points', ctypes.c_ubyte),
    ('scaling_points_cr', ctypes.c_ubyte * 2 * 10),
    ('reserved7_8bits', ctypes.c_ubyte),
    ('random_seed', ctypes.c_uint16),
    ('ar_coeffs_y', ctypes.c_int16 * 24),
    ('ar_coeffs_cb', ctypes.c_int16 * 25),
    ('ar_coeffs_cr', ctypes.c_int16 * 25),
    ('cb_mult', ctypes.c_ubyte),
    ('cb_luma_mult', ctypes.c_ubyte),
    ('cb_offset', ctypes.c_int16),
    ('cr_mult', ctypes.c_ubyte),
    ('cr_luma_mult', ctypes.c_ubyte),
    ('cr_offset', ctypes.c_int16),
    ('reserved', ctypes.c_int32 * 7),
]

union__CUVIDPICPARAMS_CodecSpecific._pack_ = 1 # source:False
union__CUVIDPICPARAMS_CodecSpecific._fields_ = [
    ('mpeg2', struct__CUVIDMPEG2PICPARAMS),
    ('h264', struct__CUVIDH264PICPARAMS),
    ('vc1', struct__CUVIDVC1PICPARAMS),
    ('mpeg4', struct__CUVIDMPEG4PICPARAMS),
    ('jpeg', struct__CUVIDJPEGPICPARAMS),
    ('hevc', struct__CUVIDHEVCPICPARAMS),
    ('vp8', struct__CUVIDVP8PICPARAMS),
    ('vp9', struct__CUVIDVP9PICPARAMS),
    ('av1', struct__CUVIDAV1PICPARAMS),
    ('CodecReserved', ctypes.c_uint32 * 1024),
]

struct__CUVIDPICPARAMS._pack_ = 1 # source:False
struct__CUVIDPICPARAMS._fields_ = [
    ('PicWidthInMbs', ctypes.c_int32),
    ('FrameHeightInMbs', ctypes.c_int32),
    ('CurrPicIdx', ctypes.c_int32),
    ('field_pic_flag', ctypes.c_int32),
    ('bottom_field_flag', ctypes.c_int32),
    ('second_field', ctypes.c_int32),
    ('nBitstreamDataLen', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pBitstreamData', ctypes.POINTER(ctypes.c_ubyte)),
    ('nNumSlices', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pSliceDataOffsets', ctypes.POINTER(ctypes.c_uint32)),
    ('ref_pic_flag', ctypes.c_int32),
    ('intra_pic_flag', ctypes.c_int32),
    ('Reserved', ctypes.c_uint32 * 30),
    ('CodecSpecific', union__CUVIDPICPARAMS_CodecSpecific),
]

CUVIDH264SVCEXT = struct__CUVIDH264SVCEXT
CUVIDH264PICPARAMS = struct__CUVIDH264PICPARAMS
CUVIDMPEG2PICPARAMS = struct__CUVIDMPEG2PICPARAMS
CUVIDMPEG4PICPARAMS = struct__CUVIDMPEG4PICPARAMS
CUVIDVC1PICPARAMS = struct__CUVIDVC1PICPARAMS
CUVIDJPEGPICPARAMS = struct__CUVIDJPEGPICPARAMS
CUVIDHEVCPICPARAMS = struct__CUVIDHEVCPICPARAMS
CUVIDVP8PICPARAMS = struct__CUVIDVP8PICPARAMS
CUVIDVP9PICPARAMS = struct__CUVIDVP9PICPARAMS
CUVIDAV1PICPARAMS = struct__CUVIDAV1PICPARAMS
CUVIDPICPARAMS = struct__CUVIDPICPARAMS
class struct__CUVIDPROCPARAMS(Structure):
    pass

class struct_CUstream_st(Structure):
    pass

struct__CUVIDPROCPARAMS._pack_ = 1 # source:False
struct__CUVIDPROCPARAMS._fields_ = [
    ('progressive_frame', ctypes.c_int32),
    ('second_field', ctypes.c_int32),
    ('top_field_first', ctypes.c_int32),
    ('unpaired_field', ctypes.c_int32),
    ('reserved_flags', ctypes.c_uint32),
    ('reserved_zero', ctypes.c_uint32),
    ('raw_input_dptr', ctypes.c_uint64),
    ('raw_input_pitch', ctypes.c_uint32),
    ('raw_input_format', ctypes.c_uint32),
    ('raw_output_dptr', ctypes.c_uint64),
    ('raw_output_pitch', ctypes.c_uint32),
    ('Reserved1', ctypes.c_uint32),
    ('output_stream', ctypes.POINTER(struct_CUstream_st)),
    ('Reserved', ctypes.c_uint32 * 46),
    ('histogram_dptr', ctypes.POINTER(ctypes.c_uint64)),
    ('Reserved2', ctypes.POINTER(None) * 1),
]

CUVIDPROCPARAMS = struct__CUVIDPROCPARAMS
class struct__CUVIDGETDECODESTATUS(Structure):
    pass

struct__CUVIDGETDECODESTATUS._pack_ = 1 # source:False
struct__CUVIDGETDECODESTATUS._fields_ = [
    ('decodeStatus', cuvidDecodeStatus),
    ('reserved', ctypes.c_uint32 * 31),
    ('pReserved', ctypes.POINTER(None) * 8),
]

CUVIDGETDECODESTATUS = struct__CUVIDGETDECODESTATUS
class struct__CUVIDRECONFIGUREDECODERINFO(Structure):
    pass

class struct__CUVIDRECONFIGUREDECODERINFO_display_area(Structure):
    pass

struct__CUVIDRECONFIGUREDECODERINFO_display_area._pack_ = 1 # source:False
struct__CUVIDRECONFIGUREDECODERINFO_display_area._fields_ = [
    ('left', ctypes.c_int16),
    ('top', ctypes.c_int16),
    ('right', ctypes.c_int16),
    ('bottom', ctypes.c_int16),
]

class struct__CUVIDRECONFIGUREDECODERINFO_target_rect(Structure):
    pass

struct__CUVIDRECONFIGUREDECODERINFO_target_rect._pack_ = 1 # source:False
struct__CUVIDRECONFIGUREDECODERINFO_target_rect._fields_ = [
    ('left', ctypes.c_int16),
    ('top', ctypes.c_int16),
    ('right', ctypes.c_int16),
    ('bottom', ctypes.c_int16),
]

struct__CUVIDRECONFIGUREDECODERINFO._pack_ = 1 # source:False
struct__CUVIDRECONFIGUREDECODERINFO._fields_ = [
    ('ulWidth', ctypes.c_uint32),
    ('ulHeight', ctypes.c_uint32),
    ('ulTargetWidth', ctypes.c_uint32),
    ('ulTargetHeight', ctypes.c_uint32),
    ('ulNumDecodeSurfaces', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32 * 12),
    ('display_area', struct__CUVIDRECONFIGUREDECODERINFO_display_area),
    ('target_rect', struct__CUVIDRECONFIGUREDECODERINFO_target_rect),
    ('reserved2', ctypes.c_uint32 * 11),
]

CUVIDRECONFIGUREDECODERINFO = struct__CUVIDRECONFIGUREDECODERINFO

# values for enumeration 'cudaError_enum'
cudaError_enum__enumvalues = {
    0: 'CUDA_SUCCESS',
    1: 'CUDA_ERROR_INVALID_VALUE',
    2: 'CUDA_ERROR_OUT_OF_MEMORY',
    3: 'CUDA_ERROR_NOT_INITIALIZED',
    4: 'CUDA_ERROR_DEINITIALIZED',
    5: 'CUDA_ERROR_PROFILER_DISABLED',
    6: 'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    7: 'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    8: 'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    34: 'CUDA_ERROR_STUB_LIBRARY',
    46: 'CUDA_ERROR_DEVICE_UNAVAILABLE',
    100: 'CUDA_ERROR_NO_DEVICE',
    101: 'CUDA_ERROR_INVALID_DEVICE',
    102: 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    200: 'CUDA_ERROR_INVALID_IMAGE',
    201: 'CUDA_ERROR_INVALID_CONTEXT',
    202: 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    205: 'CUDA_ERROR_MAP_FAILED',
    206: 'CUDA_ERROR_UNMAP_FAILED',
    207: 'CUDA_ERROR_ARRAY_IS_MAPPED',
    208: 'CUDA_ERROR_ALREADY_MAPPED',
    209: 'CUDA_ERROR_NO_BINARY_FOR_GPU',
    210: 'CUDA_ERROR_ALREADY_ACQUIRED',
    211: 'CUDA_ERROR_NOT_MAPPED',
    212: 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    213: 'CUDA_ERROR_NOT_MAPPED_AS_POINTER',
    214: 'CUDA_ERROR_ECC_UNCORRECTABLE',
    215: 'CUDA_ERROR_UNSUPPORTED_LIMIT',
    216: 'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    217: 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    218: 'CUDA_ERROR_INVALID_PTX',
    219: 'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    220: 'CUDA_ERROR_NVLINK_UNCORRECTABLE',
    221: 'CUDA_ERROR_JIT_COMPILER_NOT_FOUND',
    222: 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
    223: 'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    224: 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    225: 'CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC',
    226: 'CUDA_ERROR_CONTAINED',
    300: 'CUDA_ERROR_INVALID_SOURCE',
    301: 'CUDA_ERROR_FILE_NOT_FOUND',
    302: 'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    303: 'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    304: 'CUDA_ERROR_OPERATING_SYSTEM',
    400: 'CUDA_ERROR_INVALID_HANDLE',
    401: 'CUDA_ERROR_ILLEGAL_STATE',
    402: 'CUDA_ERROR_LOSSY_QUERY',
    500: 'CUDA_ERROR_NOT_FOUND',
    600: 'CUDA_ERROR_NOT_READY',
    700: 'CUDA_ERROR_ILLEGAL_ADDRESS',
    701: 'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES',
    702: 'CUDA_ERROR_LAUNCH_TIMEOUT',
    703: 'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    704: 'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    705: 'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    708: 'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    709: 'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    710: 'CUDA_ERROR_ASSERT',
    711: 'CUDA_ERROR_TOO_MANY_PEERS',
    712: 'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    713: 'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    714: 'CUDA_ERROR_HARDWARE_STACK_ERROR',
    715: 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    716: 'CUDA_ERROR_MISALIGNED_ADDRESS',
    717: 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    718: 'CUDA_ERROR_INVALID_PC',
    719: 'CUDA_ERROR_LAUNCH_FAILED',
    720: 'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    721: 'CUDA_ERROR_TENSOR_MEMORY_LEAK',
    800: 'CUDA_ERROR_NOT_PERMITTED',
    801: 'CUDA_ERROR_NOT_SUPPORTED',
    802: 'CUDA_ERROR_SYSTEM_NOT_READY',
    803: 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    804: 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    805: 'CUDA_ERROR_MPS_CONNECTION_FAILED',
    806: 'CUDA_ERROR_MPS_RPC_FAILURE',
    807: 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    808: 'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    809: 'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    810: 'CUDA_ERROR_MPS_CLIENT_TERMINATED',
    811: 'CUDA_ERROR_CDP_NOT_SUPPORTED',
    812: 'CUDA_ERROR_CDP_VERSION_MISMATCH',
    900: 'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    901: 'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    902: 'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    903: 'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    904: 'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    905: 'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    906: 'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    907: 'CUDA_ERROR_CAPTURED_EVENT',
    908: 'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    909: 'CUDA_ERROR_TIMEOUT',
    910: 'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    911: 'CUDA_ERROR_EXTERNAL_DEVICE',
    912: 'CUDA_ERROR_INVALID_CLUSTER_SIZE',
    913: 'CUDA_ERROR_FUNCTION_NOT_LOADED',
    914: 'CUDA_ERROR_INVALID_RESOURCE_TYPE',
    915: 'CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION',
    916: 'CUDA_ERROR_KEY_ROTATION',
    999: 'CUDA_ERROR_UNKNOWN',
}
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_PROFILER_DISABLED = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
CUDA_ERROR_STUB_LIBRARY = 34
CUDA_ERROR_DEVICE_UNAVAILABLE = 46
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_DEVICE_NOT_LICENSED = 102
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED = 205
CUDA_ERROR_UNMAP_FAILED = 206
CUDA_ERROR_ARRAY_IS_MAPPED = 207
CUDA_ERROR_ALREADY_MAPPED = 208
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_ALREADY_ACQUIRED = 210
CUDA_ERROR_NOT_MAPPED = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
CUDA_ERROR_ECC_UNCORRECTABLE = 214
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225
CUDA_ERROR_CONTAINED = 226
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_FILE_NOT_FOUND = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
CUDA_ERROR_OPERATING_SYSTEM = 304
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_ILLEGAL_STATE = 401
CUDA_ERROR_LOSSY_QUERY = 402
CUDA_ERROR_NOT_FOUND = 500
CUDA_ERROR_NOT_READY = 600
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_TOO_MANY_PEERS = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
CUDA_ERROR_INVALID_PC = 718
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
CUDA_ERROR_TENSOR_MEMORY_LEAK = 721
CUDA_ERROR_NOT_PERMITTED = 800
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_SYSTEM_NOT_READY = 802
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
CUDA_ERROR_MPS_CONNECTION_FAILED = 805
CUDA_ERROR_MPS_RPC_FAILURE = 806
CUDA_ERROR_MPS_SERVER_NOT_READY = 807
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
CUDA_ERROR_MPS_CLIENT_TERMINATED = 810
CUDA_ERROR_CDP_NOT_SUPPORTED = 811
CUDA_ERROR_CDP_VERSION_MISMATCH = 812
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
CUDA_ERROR_CAPTURED_EVENT = 907
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
CUDA_ERROR_TIMEOUT = 909
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
CUDA_ERROR_EXTERNAL_DEVICE = 911
CUDA_ERROR_INVALID_CLUSTER_SIZE = 912
CUDA_ERROR_FUNCTION_NOT_LOADED = 913
CUDA_ERROR_INVALID_RESOURCE_TYPE = 914
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915
CUDA_ERROR_KEY_ROTATION = 916
CUDA_ERROR_UNKNOWN = 999
cudaError_enum = ctypes.c_uint32 # enum
CUresult = cudaError_enum
CUresult__enumvalues = cudaError_enum__enumvalues
try:
    cuvidGetDecoderCaps = _libraries['libnvcuvid.so'].cuvidGetDecoderCaps
    cuvidGetDecoderCaps.restype = CUresult
    cuvidGetDecoderCaps.argtypes = [ctypes.POINTER(struct__CUVIDDECODECAPS)]
except AttributeError:
    pass
try:
    cuvidCreateDecoder = _libraries['libnvcuvid.so'].cuvidCreateDecoder
    cuvidCreateDecoder.restype = CUresult
    cuvidCreateDecoder.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct__CUVIDDECODECREATEINFO)]
except AttributeError:
    pass
try:
    cuvidDestroyDecoder = _libraries['libnvcuvid.so'].cuvidDestroyDecoder
    cuvidDestroyDecoder.restype = CUresult
    cuvidDestroyDecoder.argtypes = [CUvideodecoder]
except AttributeError:
    pass
try:
    cuvidDecodePicture = _libraries['libnvcuvid.so'].cuvidDecodePicture
    cuvidDecodePicture.restype = CUresult
    cuvidDecodePicture.argtypes = [CUvideodecoder, ctypes.POINTER(struct__CUVIDPICPARAMS)]
except AttributeError:
    pass
try:
    cuvidGetDecodeStatus = _libraries['libnvcuvid.so'].cuvidGetDecodeStatus
    cuvidGetDecodeStatus.restype = CUresult
    cuvidGetDecodeStatus.argtypes = [CUvideodecoder, ctypes.c_int32, ctypes.POINTER(struct__CUVIDGETDECODESTATUS)]
except AttributeError:
    pass
try:
    cuvidReconfigureDecoder = _libraries['libnvcuvid.so'].cuvidReconfigureDecoder
    cuvidReconfigureDecoder.restype = CUresult
    cuvidReconfigureDecoder.argtypes = [CUvideodecoder, ctypes.POINTER(struct__CUVIDRECONFIGUREDECODERINFO)]
except AttributeError:
    pass
try:
    cuvidMapVideoFrame = _libraries['libnvcuvid.so'].cuvidMapVideoFrame
    cuvidMapVideoFrame.restype = CUresult
    cuvidMapVideoFrame.argtypes = [CUvideodecoder, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__CUVIDPROCPARAMS)]
except AttributeError:
    pass
try:
    cuvidUnmapVideoFrame = _libraries['libnvcuvid.so'].cuvidUnmapVideoFrame
    cuvidUnmapVideoFrame.restype = CUresult
    cuvidUnmapVideoFrame.argtypes = [CUvideodecoder, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuvidMapVideoFrame64 = _libraries['libnvcuvid.so'].cuvidMapVideoFrame64
    cuvidMapVideoFrame64.restype = CUresult
    cuvidMapVideoFrame64.argtypes = [CUvideodecoder, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(struct__CUVIDPROCPARAMS)]
except AttributeError:
    pass
try:
    cuvidUnmapVideoFrame64 = _libraries['libnvcuvid.so'].cuvidUnmapVideoFrame64
    cuvidUnmapVideoFrame64.restype = CUresult
    cuvidUnmapVideoFrame64.argtypes = [CUvideodecoder, ctypes.c_uint64]
except AttributeError:
    pass
class struct_CUctx_st(Structure):
    pass

CUcontext = ctypes.POINTER(struct_CUctx_st)
try:
    cuvidCtxLockCreate = _libraries['libnvcuvid.so'].cuvidCtxLockCreate
    cuvidCtxLockCreate.restype = CUresult
    cuvidCtxLockCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct__CUcontextlock_st)), CUcontext]
except AttributeError:
    pass
try:
    cuvidCtxLockDestroy = _libraries['libnvcuvid.so'].cuvidCtxLockDestroy
    cuvidCtxLockDestroy.restype = CUresult
    cuvidCtxLockDestroy.argtypes = [CUvideoctxlock]
except AttributeError:
    pass
try:
    cuvidCtxLock = _libraries['libnvcuvid.so'].cuvidCtxLock
    cuvidCtxLock.restype = CUresult
    cuvidCtxLock.argtypes = [CUvideoctxlock, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuvidCtxUnlock = _libraries['libnvcuvid.so'].cuvidCtxUnlock
    cuvidCtxUnlock.restype = CUresult
    cuvidCtxUnlock.argtypes = [CUvideoctxlock, ctypes.c_uint32]
except AttributeError:
    pass
CUvideosource = ctypes.POINTER(None)
CUvideoparser = ctypes.POINTER(None)
CUvideotimestamp = ctypes.c_int64

# values for enumeration 'c__EA_cudaVideoState'
c__EA_cudaVideoState__enumvalues = {
    -1: 'cudaVideoState_Error',
    0: 'cudaVideoState_Stopped',
    1: 'cudaVideoState_Started',
}
cudaVideoState_Error = -1
cudaVideoState_Stopped = 0
cudaVideoState_Started = 1
c__EA_cudaVideoState = ctypes.c_int32 # enum
cudaVideoState = c__EA_cudaVideoState
cudaVideoState__enumvalues = c__EA_cudaVideoState__enumvalues

# values for enumeration 'c__EA_cudaAudioCodec'
c__EA_cudaAudioCodec__enumvalues = {
    0: 'cudaAudioCodec_MPEG1',
    1: 'cudaAudioCodec_MPEG2',
    2: 'cudaAudioCodec_MP3',
    3: 'cudaAudioCodec_AC3',
    4: 'cudaAudioCodec_LPCM',
    5: 'cudaAudioCodec_AAC',
}
cudaAudioCodec_MPEG1 = 0
cudaAudioCodec_MPEG2 = 1
cudaAudioCodec_MP3 = 2
cudaAudioCodec_AC3 = 3
cudaAudioCodec_LPCM = 4
cudaAudioCodec_AAC = 5
c__EA_cudaAudioCodec = ctypes.c_uint32 # enum
cudaAudioCodec = c__EA_cudaAudioCodec
cudaAudioCodec__enumvalues = c__EA_cudaAudioCodec__enumvalues
class struct__TIMECODESET(Structure):
    pass

struct__TIMECODESET._pack_ = 1 # source:False
struct__TIMECODESET._fields_ = [
    ('time_offset_value', ctypes.c_uint32),
    ('n_frames', ctypes.c_uint16),
    ('clock_timestamp_flag', ctypes.c_ubyte),
    ('units_field_based_flag', ctypes.c_ubyte),
    ('counting_type', ctypes.c_ubyte),
    ('full_timestamp_flag', ctypes.c_ubyte),
    ('discontinuity_flag', ctypes.c_ubyte),
    ('cnt_dropped_flag', ctypes.c_ubyte),
    ('seconds_value', ctypes.c_ubyte),
    ('minutes_value', ctypes.c_ubyte),
    ('hours_value', ctypes.c_ubyte),
    ('seconds_flag', ctypes.c_ubyte),
    ('minutes_flag', ctypes.c_ubyte),
    ('hours_flag', ctypes.c_ubyte),
    ('time_offset_length', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

TIMECODESET = struct__TIMECODESET
class struct__TIMECODE(Structure):
    pass

struct__TIMECODE._pack_ = 1 # source:False
struct__TIMECODE._fields_ = [
    ('time_code_set', struct__TIMECODESET * 3),
    ('num_clock_ts', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

TIMECODE = struct__TIMECODE
class struct__SEIMASTERINGDISPLAYINFO(Structure):
    pass

struct__SEIMASTERINGDISPLAYINFO._pack_ = 1 # source:False
struct__SEIMASTERINGDISPLAYINFO._fields_ = [
    ('display_primaries_x', ctypes.c_uint16 * 3),
    ('display_primaries_y', ctypes.c_uint16 * 3),
    ('white_point_x', ctypes.c_uint16),
    ('white_point_y', ctypes.c_uint16),
    ('max_display_mastering_luminance', ctypes.c_uint32),
    ('min_display_mastering_luminance', ctypes.c_uint32),
]

SEIMASTERINGDISPLAYINFO = struct__SEIMASTERINGDISPLAYINFO
class struct__SEICONTENTLIGHTLEVELINFO(Structure):
    pass

struct__SEICONTENTLIGHTLEVELINFO._pack_ = 1 # source:False
struct__SEICONTENTLIGHTLEVELINFO._fields_ = [
    ('max_content_light_level', ctypes.c_uint16),
    ('max_pic_average_light_level', ctypes.c_uint16),
    ('reserved', ctypes.c_uint32),
]

SEICONTENTLIGHTLEVELINFO = struct__SEICONTENTLIGHTLEVELINFO
class struct__TIMECODEMPEG2(Structure):
    pass

struct__TIMECODEMPEG2._pack_ = 1 # source:False
struct__TIMECODEMPEG2._fields_ = [
    ('drop_frame_flag', ctypes.c_ubyte),
    ('time_code_hours', ctypes.c_ubyte),
    ('time_code_minutes', ctypes.c_ubyte),
    ('marker_bit', ctypes.c_ubyte),
    ('time_code_seconds', ctypes.c_ubyte),
    ('time_code_pictures', ctypes.c_ubyte),
]

TIMECODEMPEG2 = struct__TIMECODEMPEG2
class struct__SEIALTERNATIVETRANSFERCHARACTERISTICS(Structure):
    pass

struct__SEIALTERNATIVETRANSFERCHARACTERISTICS._pack_ = 1 # source:False
struct__SEIALTERNATIVETRANSFERCHARACTERISTICS._fields_ = [
    ('preferred_transfer_characteristics', ctypes.c_ubyte),
]

SEIALTERNATIVETRANSFERCHARACTERISTICS = struct__SEIALTERNATIVETRANSFERCHARACTERISTICS
class struct__CUSEIMESSAGE(Structure):
    pass

struct__CUSEIMESSAGE._pack_ = 1 # source:False
struct__CUSEIMESSAGE._fields_ = [
    ('sei_message_type', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte * 3),
    ('sei_message_size', ctypes.c_uint32),
]

CUSEIMESSAGE = struct__CUSEIMESSAGE
class struct_c__SA_CUVIDEOFORMAT(Structure):
    pass

class struct_c__SA_CUVIDEOFORMAT_frame_rate(Structure):
    pass

struct_c__SA_CUVIDEOFORMAT_frame_rate._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMAT_frame_rate._fields_ = [
    ('numerator', ctypes.c_uint32),
    ('denominator', ctypes.c_uint32),
]

class struct_c__SA_CUVIDEOFORMAT_display_area(Structure):
    pass

struct_c__SA_CUVIDEOFORMAT_display_area._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMAT_display_area._fields_ = [
    ('left', ctypes.c_int32),
    ('top', ctypes.c_int32),
    ('right', ctypes.c_int32),
    ('bottom', ctypes.c_int32),
]

class struct_c__SA_CUVIDEOFORMAT_display_aspect_ratio(Structure):
    pass

struct_c__SA_CUVIDEOFORMAT_display_aspect_ratio._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMAT_display_aspect_ratio._fields_ = [
    ('x', ctypes.c_int32),
    ('y', ctypes.c_int32),
]

class struct_c__SA_CUVIDEOFORMAT_video_signal_description(Structure):
    pass

struct_c__SA_CUVIDEOFORMAT_video_signal_description._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMAT_video_signal_description._fields_ = [
    ('video_format', ctypes.c_ubyte, 3),
    ('video_full_range_flag', ctypes.c_ubyte, 1),
    ('reserved_zero_bits', ctypes.c_ubyte, 4),
    ('color_primaries', ctypes.c_ubyte, 8),
    ('transfer_characteristics', ctypes.c_ubyte),
    ('matrix_coefficients', ctypes.c_ubyte),
]

struct_c__SA_CUVIDEOFORMAT._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMAT._fields_ = [
    ('codec', cudaVideoCodec),
    ('frame_rate', struct_c__SA_CUVIDEOFORMAT_frame_rate),
    ('progressive_sequence', ctypes.c_ubyte),
    ('bit_depth_luma_minus8', ctypes.c_ubyte),
    ('bit_depth_chroma_minus8', ctypes.c_ubyte),
    ('min_num_decode_surfaces', ctypes.c_ubyte),
    ('coded_width', ctypes.c_uint32),
    ('coded_height', ctypes.c_uint32),
    ('display_area', struct_c__SA_CUVIDEOFORMAT_display_area),
    ('chroma_format', cudaVideoChromaFormat),
    ('bitrate', ctypes.c_uint32),
    ('display_aspect_ratio', struct_c__SA_CUVIDEOFORMAT_display_aspect_ratio),
    ('video_signal_description', struct_c__SA_CUVIDEOFORMAT_video_signal_description),
    ('seqhdr_data_length', ctypes.c_uint32),
]

CUVIDEOFORMAT = struct_c__SA_CUVIDEOFORMAT
class struct_c__SA_CUVIDOPERATINGPOINTINFO(Structure):
    pass

class union_c__SA_CUVIDOPERATINGPOINTINFO_0(Union):
    pass

class struct_c__SA_CUVIDOPERATINGPOINTINFO_0_av1(Structure):
    pass

struct_c__SA_CUVIDOPERATINGPOINTINFO_0_av1._pack_ = 1 # source:False
struct_c__SA_CUVIDOPERATINGPOINTINFO_0_av1._fields_ = [
    ('operating_points_cnt', ctypes.c_ubyte),
    ('reserved24_bits', ctypes.c_ubyte * 3),
    ('operating_points_idc', ctypes.c_uint16 * 32),
]

union_c__SA_CUVIDOPERATINGPOINTINFO_0._pack_ = 1 # source:False
union_c__SA_CUVIDOPERATINGPOINTINFO_0._fields_ = [
    ('av1', struct_c__SA_CUVIDOPERATINGPOINTINFO_0_av1),
    ('CodecReserved', ctypes.c_ubyte * 1024),
]

struct_c__SA_CUVIDOPERATINGPOINTINFO._pack_ = 1 # source:False
struct_c__SA_CUVIDOPERATINGPOINTINFO._anonymous_ = ('_0',)
struct_c__SA_CUVIDOPERATINGPOINTINFO._fields_ = [
    ('codec', cudaVideoCodec),
    ('_0', union_c__SA_CUVIDOPERATINGPOINTINFO_0),
]

CUVIDOPERATINGPOINTINFO = struct_c__SA_CUVIDOPERATINGPOINTINFO
class struct__CUVIDSEIMESSAGEINFO(Structure):
    pass

struct__CUVIDSEIMESSAGEINFO._pack_ = 1 # source:False
struct__CUVIDSEIMESSAGEINFO._fields_ = [
    ('pSEIData', ctypes.POINTER(None)),
    ('pSEIMessage', ctypes.POINTER(struct__CUSEIMESSAGE)),
    ('sei_message_count', ctypes.c_uint32),
    ('picIdx', ctypes.c_uint32),
]

CUVIDSEIMESSAGEINFO = struct__CUVIDSEIMESSAGEINFO
class struct_c__SA_CUVIDAV1SEQHDR(Structure):
    pass

struct_c__SA_CUVIDAV1SEQHDR._pack_ = 1 # source:False
struct_c__SA_CUVIDAV1SEQHDR._fields_ = [
    ('max_width', ctypes.c_uint32),
    ('max_height', ctypes.c_uint32),
    ('reserved', ctypes.c_ubyte * 1016),
]

CUVIDAV1SEQHDR = struct_c__SA_CUVIDAV1SEQHDR
class struct_c__SA_CUVIDEOFORMATEX(Structure):
    pass

class union_c__SA_CUVIDEOFORMATEX_0(Union):
    pass

union_c__SA_CUVIDEOFORMATEX_0._pack_ = 1 # source:False
union_c__SA_CUVIDEOFORMATEX_0._fields_ = [
    ('av1', CUVIDAV1SEQHDR),
    ('raw_seqhdr_data', ctypes.c_ubyte * 1024),
]

struct_c__SA_CUVIDEOFORMATEX._pack_ = 1 # source:False
struct_c__SA_CUVIDEOFORMATEX._anonymous_ = ('_0',)
struct_c__SA_CUVIDEOFORMATEX._fields_ = [
    ('format', CUVIDEOFORMAT),
    ('_0', union_c__SA_CUVIDEOFORMATEX_0),
]

CUVIDEOFORMATEX = struct_c__SA_CUVIDEOFORMATEX
class struct_c__SA_CUAUDIOFORMAT(Structure):
    pass

struct_c__SA_CUAUDIOFORMAT._pack_ = 1 # source:False
struct_c__SA_CUAUDIOFORMAT._fields_ = [
    ('codec', cudaAudioCodec),
    ('channels', ctypes.c_uint32),
    ('samplespersec', ctypes.c_uint32),
    ('bitrate', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint32),
]

CUAUDIOFORMAT = struct_c__SA_CUAUDIOFORMAT

# values for enumeration 'c__EA_CUvideopacketflags'
c__EA_CUvideopacketflags__enumvalues = {
    1: 'CUVID_PKT_ENDOFSTREAM',
    2: 'CUVID_PKT_TIMESTAMP',
    4: 'CUVID_PKT_DISCONTINUITY',
    8: 'CUVID_PKT_ENDOFPICTURE',
    16: 'CUVID_PKT_NOTIFY_EOS',
}
CUVID_PKT_ENDOFSTREAM = 1
CUVID_PKT_TIMESTAMP = 2
CUVID_PKT_DISCONTINUITY = 4
CUVID_PKT_ENDOFPICTURE = 8
CUVID_PKT_NOTIFY_EOS = 16
c__EA_CUvideopacketflags = ctypes.c_uint32 # enum
CUvideopacketflags = c__EA_CUvideopacketflags
CUvideopacketflags__enumvalues = c__EA_CUvideopacketflags__enumvalues
class struct__CUVIDSOURCEDATAPACKET(Structure):
    pass

struct__CUVIDSOURCEDATAPACKET._pack_ = 1 # source:False
struct__CUVIDSOURCEDATAPACKET._fields_ = [
    ('flags', ctypes.c_uint64),
    ('payload_size', ctypes.c_uint64),
    ('payload', ctypes.POINTER(ctypes.c_ubyte)),
    ('timestamp', ctypes.c_int64),
]

CUVIDSOURCEDATAPACKET = struct__CUVIDSOURCEDATAPACKET
PFNVIDSOURCECALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDSOURCEDATAPACKET))
class struct__CUVIDSOURCEPARAMS(Structure):
    pass

struct__CUVIDSOURCEPARAMS._pack_ = 1 # source:False
struct__CUVIDSOURCEPARAMS._fields_ = [
    ('ulClockRate', ctypes.c_uint32),
    ('bAnnexb', ctypes.c_uint32, 1),
    ('uReserved', ctypes.c_uint32, 31),
    ('uReserved1', ctypes.c_uint32 * 6),
    ('pUserData', ctypes.POINTER(None)),
    ('pfnVideoDataHandler', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDSOURCEDATAPACKET))),
    ('pfnAudioDataHandler', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDSOURCEDATAPACKET))),
    ('pvReserved2', ctypes.POINTER(None) * 8),
]

CUVIDSOURCEPARAMS = struct__CUVIDSOURCEPARAMS

# values for enumeration 'c__EA_CUvideosourceformat_flags'
c__EA_CUvideosourceformat_flags__enumvalues = {
    256: 'CUVID_FMT_EXTFORMATINFO',
}
CUVID_FMT_EXTFORMATINFO = 256
c__EA_CUvideosourceformat_flags = ctypes.c_uint32 # enum
CUvideosourceformat_flags = c__EA_CUvideosourceformat_flags
CUvideosourceformat_flags__enumvalues = c__EA_CUvideosourceformat_flags__enumvalues
try:
    cuvidCreateVideoSource = _libraries['libnvcuvid.so'].cuvidCreateVideoSource
    cuvidCreateVideoSource.restype = CUresult
    cuvidCreateVideoSource.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct__CUVIDSOURCEPARAMS)]
except AttributeError:
    pass
try:
    cuvidCreateVideoSourceW = _libraries['libnvcuvid.so'].cuvidCreateVideoSourceW
    cuvidCreateVideoSourceW.restype = CUresult
    cuvidCreateVideoSourceW.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(struct__CUVIDSOURCEPARAMS)]
except AttributeError:
    pass
try:
    cuvidDestroyVideoSource = _libraries['libnvcuvid.so'].cuvidDestroyVideoSource
    cuvidDestroyVideoSource.restype = CUresult
    cuvidDestroyVideoSource.argtypes = [CUvideosource]
except AttributeError:
    pass
try:
    cuvidSetVideoSourceState = _libraries['libnvcuvid.so'].cuvidSetVideoSourceState
    cuvidSetVideoSourceState.restype = CUresult
    cuvidSetVideoSourceState.argtypes = [CUvideosource, cudaVideoState]
except AttributeError:
    pass
try:
    cuvidGetVideoSourceState = _libraries['libnvcuvid.so'].cuvidGetVideoSourceState
    cuvidGetVideoSourceState.restype = cudaVideoState
    cuvidGetVideoSourceState.argtypes = [CUvideosource]
except AttributeError:
    pass
try:
    cuvidGetSourceVideoFormat = _libraries['libnvcuvid.so'].cuvidGetSourceVideoFormat
    cuvidGetSourceVideoFormat.restype = CUresult
    cuvidGetSourceVideoFormat.argtypes = [CUvideosource, ctypes.POINTER(struct_c__SA_CUVIDEOFORMAT), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuvidGetSourceAudioFormat = _libraries['libnvcuvid.so'].cuvidGetSourceAudioFormat
    cuvidGetSourceAudioFormat.restype = CUresult
    cuvidGetSourceAudioFormat.argtypes = [CUvideosource, ctypes.POINTER(struct_c__SA_CUAUDIOFORMAT), ctypes.c_uint32]
except AttributeError:
    pass
class struct__CUVIDPARSERDISPINFO(Structure):
    pass

struct__CUVIDPARSERDISPINFO._pack_ = 1 # source:False
struct__CUVIDPARSERDISPINFO._fields_ = [
    ('picture_index', ctypes.c_int32),
    ('progressive_frame', ctypes.c_int32),
    ('top_field_first', ctypes.c_int32),
    ('repeat_first_field', ctypes.c_int32),
    ('timestamp', ctypes.c_int64),
]

CUVIDPARSERDISPINFO = struct__CUVIDPARSERDISPINFO
PFNVIDSEQUENCECALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_CUVIDEOFORMAT))
PFNVIDDECODECALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDPICPARAMS))
PFNVIDDISPLAYCALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDPARSERDISPINFO))
PFNVIDOPPOINTCALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_CUVIDOPERATINGPOINTINFO))
PFNVIDSEIMSGCALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDSEIMESSAGEINFO))
class struct__CUVIDPARSERPARAMS(Structure):
    pass

struct__CUVIDPARSERPARAMS._pack_ = 1 # source:False
struct__CUVIDPARSERPARAMS._fields_ = [
    ('CodecType', cudaVideoCodec),
    ('ulMaxNumDecodeSurfaces', ctypes.c_uint32),
    ('ulClockRate', ctypes.c_uint32),
    ('ulErrorThreshold', ctypes.c_uint32),
    ('ulMaxDisplayDelay', ctypes.c_uint32),
    ('bAnnexb', ctypes.c_uint32, 1),
    ('bMemoryOptimize', ctypes.c_uint32, 1),
    ('uReserved', ctypes.c_uint32, 30),
    ('uReserved1', ctypes.c_uint32 * 4),
    ('pUserData', ctypes.POINTER(None)),
    ('pfnSequenceCallback', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_CUVIDEOFORMAT))),
    ('pfnDecodePicture', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDPICPARAMS))),
    ('pfnDisplayPicture', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDPARSERDISPINFO))),
    ('pfnGetOperatingPoint', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct_c__SA_CUVIDOPERATINGPOINTINFO))),
    ('pfnGetSEIMsg', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(struct__CUVIDSEIMESSAGEINFO))),
    ('pvReserved2', ctypes.POINTER(None) * 5),
    ('pExtVideoInfo', ctypes.POINTER(struct_c__SA_CUVIDEOFORMATEX)),
]

CUVIDPARSERPARAMS = struct__CUVIDPARSERPARAMS
try:
    cuvidCreateVideoParser = _libraries['libnvcuvid.so'].cuvidCreateVideoParser
    cuvidCreateVideoParser.restype = CUresult
    cuvidCreateVideoParser.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct__CUVIDPARSERPARAMS)]
except AttributeError:
    pass
try:
    cuvidParseVideoData = _libraries['libnvcuvid.so'].cuvidParseVideoData
    cuvidParseVideoData.restype = CUresult
    cuvidParseVideoData.argtypes = [CUvideoparser, ctypes.POINTER(struct__CUVIDSOURCEDATAPACKET)]
except AttributeError:
    pass
try:
    cuvidDestroyVideoParser = _libraries['libnvcuvid.so'].cuvidDestroyVideoParser
    cuvidDestroyVideoParser.restype = CUresult
    cuvidDestroyVideoParser.argtypes = [CUvideoparser]
except AttributeError:
    pass
__all__ = \
    ['CUAUDIOFORMAT', 'CUDA_ERROR_ALREADY_ACQUIRED',
    'CUDA_ERROR_ALREADY_MAPPED', 'CUDA_ERROR_ARRAY_IS_MAPPED',
    'CUDA_ERROR_ASSERT', 'CUDA_ERROR_CAPTURED_EVENT',
    'CUDA_ERROR_CDP_NOT_SUPPORTED', 'CUDA_ERROR_CDP_VERSION_MISMATCH',
    'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    'CUDA_ERROR_CONTAINED', 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    'CUDA_ERROR_DEINITIALIZED', 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    'CUDA_ERROR_DEVICE_UNAVAILABLE', 'CUDA_ERROR_ECC_UNCORRECTABLE',
    'CUDA_ERROR_EXTERNAL_DEVICE', 'CUDA_ERROR_FILE_NOT_FOUND',
    'CUDA_ERROR_FUNCTION_NOT_LOADED',
    'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    'CUDA_ERROR_HARDWARE_STACK_ERROR',
    'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    'CUDA_ERROR_ILLEGAL_ADDRESS', 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    'CUDA_ERROR_ILLEGAL_STATE', 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    'CUDA_ERROR_INVALID_CLUSTER_SIZE', 'CUDA_ERROR_INVALID_CONTEXT',
    'CUDA_ERROR_INVALID_DEVICE',
    'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    'CUDA_ERROR_INVALID_HANDLE', 'CUDA_ERROR_INVALID_IMAGE',
    'CUDA_ERROR_INVALID_PC', 'CUDA_ERROR_INVALID_PTX',
    'CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION',
    'CUDA_ERROR_INVALID_RESOURCE_TYPE', 'CUDA_ERROR_INVALID_SOURCE',
    'CUDA_ERROR_INVALID_VALUE', 'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    'CUDA_ERROR_JIT_COMPILER_NOT_FOUND', 'CUDA_ERROR_KEY_ROTATION',
    'CUDA_ERROR_LAUNCH_FAILED',
    'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES', 'CUDA_ERROR_LAUNCH_TIMEOUT',
    'CUDA_ERROR_LOSSY_QUERY', 'CUDA_ERROR_MAP_FAILED',
    'CUDA_ERROR_MISALIGNED_ADDRESS',
    'CUDA_ERROR_MPS_CLIENT_TERMINATED',
    'CUDA_ERROR_MPS_CONNECTION_FAILED',
    'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    'CUDA_ERROR_MPS_RPC_FAILURE', 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    'CUDA_ERROR_NOT_FOUND', 'CUDA_ERROR_NOT_INITIALIZED',
    'CUDA_ERROR_NOT_MAPPED', 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    'CUDA_ERROR_NOT_MAPPED_AS_POINTER', 'CUDA_ERROR_NOT_PERMITTED',
    'CUDA_ERROR_NOT_READY', 'CUDA_ERROR_NOT_SUPPORTED',
    'CUDA_ERROR_NO_BINARY_FOR_GPU', 'CUDA_ERROR_NO_DEVICE',
    'CUDA_ERROR_NVLINK_UNCORRECTABLE', 'CUDA_ERROR_OPERATING_SYSTEM',
    'CUDA_ERROR_OUT_OF_MEMORY',
    'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    'CUDA_ERROR_PROFILER_DISABLED',
    'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    'CUDA_ERROR_STUB_LIBRARY', 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    'CUDA_ERROR_SYSTEM_NOT_READY', 'CUDA_ERROR_TENSOR_MEMORY_LEAK',
    'CUDA_ERROR_TIMEOUT', 'CUDA_ERROR_TOO_MANY_PEERS',
    'CUDA_ERROR_UNKNOWN', 'CUDA_ERROR_UNMAP_FAILED',
    'CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC',
    'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    'CUDA_ERROR_UNSUPPORTED_LIMIT',
    'CUDA_ERROR_UNSUPPORTED_PTX_VERSION', 'CUDA_SUCCESS',
    'CUSEIMESSAGE', 'CUVIDAV1PICPARAMS', 'CUVIDAV1SEQHDR',
    'CUVIDDECODECAPS', 'CUVIDDECODECREATEINFO', 'CUVIDEOFORMAT',
    'CUVIDEOFORMATEX', 'CUVIDGETDECODESTATUS', 'CUVIDH264DPBENTRY',
    'CUVIDH264MVCEXT', 'CUVIDH264PICPARAMS', 'CUVIDH264SVCEXT',
    'CUVIDHEVCPICPARAMS', 'CUVIDJPEGPICPARAMS', 'CUVIDMPEG2PICPARAMS',
    'CUVIDMPEG4PICPARAMS', 'CUVIDOPERATINGPOINTINFO',
    'CUVIDPARSERDISPINFO', 'CUVIDPARSERPARAMS', 'CUVIDPICPARAMS',
    'CUVIDPROCPARAMS', 'CUVIDRECONFIGUREDECODERINFO',
    'CUVIDSEIMESSAGEINFO', 'CUVIDSOURCEDATAPACKET',
    'CUVIDSOURCEPARAMS', 'CUVIDVC1PICPARAMS', 'CUVIDVP8PICPARAMS',
    'CUVIDVP9PICPARAMS', 'CUVID_FMT_EXTFORMATINFO',
    'CUVID_PKT_DISCONTINUITY', 'CUVID_PKT_ENDOFPICTURE',
    'CUVID_PKT_ENDOFSTREAM', 'CUVID_PKT_NOTIFY_EOS',
    'CUVID_PKT_TIMESTAMP', 'CUcontext', 'CUresult',
    'CUresult__enumvalues', 'CUvideoctxlock', 'CUvideodecoder',
    'CUvideopacketflags', 'CUvideopacketflags__enumvalues',
    'CUvideoparser', 'CUvideosource', 'CUvideosourceformat_flags',
    'CUvideosourceformat_flags__enumvalues', 'CUvideotimestamp',
    'PFNVIDDECODECALLBACK', 'PFNVIDDISPLAYCALLBACK',
    'PFNVIDOPPOINTCALLBACK', 'PFNVIDSEIMSGCALLBACK',
    'PFNVIDSEQUENCECALLBACK', 'PFNVIDSOURCECALLBACK',
    'SEIALTERNATIVETRANSFERCHARACTERISTICS',
    'SEICONTENTLIGHTLEVELINFO', 'SEIMASTERINGDISPLAYINFO', 'TIMECODE',
    'TIMECODEMPEG2', 'TIMECODESET', 'c__EA_CUvideopacketflags',
    'c__EA_CUvideosourceformat_flags', 'c__EA_cudaAudioCodec',
    'c__EA_cudaVideoState', 'cudaAudioCodec', 'cudaAudioCodec_AAC',
    'cudaAudioCodec_AC3', 'cudaAudioCodec_LPCM', 'cudaAudioCodec_MP3',
    'cudaAudioCodec_MPEG1', 'cudaAudioCodec_MPEG2',
    'cudaAudioCodec__enumvalues', 'cudaError_enum',
    'cudaVideoChromaFormat', 'cudaVideoChromaFormat_420',
    'cudaVideoChromaFormat_422', 'cudaVideoChromaFormat_444',
    'cudaVideoChromaFormat_Monochrome',
    'cudaVideoChromaFormat__enumvalues', 'cudaVideoChromaFormat_enum',
    'cudaVideoCodec', 'cudaVideoCodec_AV1', 'cudaVideoCodec_H264',
    'cudaVideoCodec_H264_MVC', 'cudaVideoCodec_H264_SVC',
    'cudaVideoCodec_HEVC', 'cudaVideoCodec_JPEG',
    'cudaVideoCodec_MPEG1', 'cudaVideoCodec_MPEG2',
    'cudaVideoCodec_MPEG4', 'cudaVideoCodec_NV12',
    'cudaVideoCodec_NumCodecs', 'cudaVideoCodec_UYVY',
    'cudaVideoCodec_VC1', 'cudaVideoCodec_VP8', 'cudaVideoCodec_VP9',
    'cudaVideoCodec_YUV420', 'cudaVideoCodec_YUYV',
    'cudaVideoCodec_YV12', 'cudaVideoCodec__enumvalues',
    'cudaVideoCodec_enum', 'cudaVideoCreateFlags',
    'cudaVideoCreateFlags__enumvalues', 'cudaVideoCreateFlags_enum',
    'cudaVideoCreate_Default', 'cudaVideoCreate_PreferCUDA',
    'cudaVideoCreate_PreferCUVID', 'cudaVideoCreate_PreferDXVA',
    'cudaVideoDeinterlaceMode', 'cudaVideoDeinterlaceMode_Adaptive',
    'cudaVideoDeinterlaceMode_Bob', 'cudaVideoDeinterlaceMode_Weave',
    'cudaVideoDeinterlaceMode__enumvalues',
    'cudaVideoDeinterlaceMode_enum', 'cudaVideoState',
    'cudaVideoState_Error', 'cudaVideoState_Started',
    'cudaVideoState_Stopped', 'cudaVideoState__enumvalues',
    'cudaVideoSurfaceFormat', 'cudaVideoSurfaceFormat_NV12',
    'cudaVideoSurfaceFormat_NV16', 'cudaVideoSurfaceFormat_P016',
    'cudaVideoSurfaceFormat_P216', 'cudaVideoSurfaceFormat_YUV444',
    'cudaVideoSurfaceFormat_YUV444_16Bit',
    'cudaVideoSurfaceFormat__enumvalues',
    'cudaVideoSurfaceFormat_enum', 'cuvidCreateDecoder',
    'cuvidCreateVideoParser', 'cuvidCreateVideoSource',
    'cuvidCreateVideoSourceW', 'cuvidCtxLock', 'cuvidCtxLockCreate',
    'cuvidCtxLockDestroy', 'cuvidCtxUnlock', 'cuvidDecodePicture',
    'cuvidDecodeStatus', 'cuvidDecodeStatus_Error',
    'cuvidDecodeStatus_Error_Concealed',
    'cuvidDecodeStatus_InProgress', 'cuvidDecodeStatus_Invalid',
    'cuvidDecodeStatus_Success', 'cuvidDecodeStatus__enumvalues',
    'cuvidDecodeStatus_enum', 'cuvidDestroyDecoder',
    'cuvidDestroyVideoParser', 'cuvidDestroyVideoSource',
    'cuvidGetDecodeStatus', 'cuvidGetDecoderCaps',
    'cuvidGetSourceAudioFormat', 'cuvidGetSourceVideoFormat',
    'cuvidGetVideoSourceState', 'cuvidMapVideoFrame',
    'cuvidMapVideoFrame64', 'cuvidParseVideoData',
    'cuvidReconfigureDecoder', 'cuvidSetVideoSourceState',
    'cuvidUnmapVideoFrame', 'cuvidUnmapVideoFrame64',
    'struct_CUctx_st', 'struct_CUstream_st', 'struct__CUSEIMESSAGE',
    'struct__CUVIDAV1PICPARAMS', 'struct__CUVIDAV1PICPARAMS_0',
    'struct__CUVIDAV1PICPARAMS_1', 'struct__CUVIDDECODECAPS',
    'struct__CUVIDDECODECREATEINFO',
    'struct__CUVIDDECODECREATEINFO_display_area',
    'struct__CUVIDDECODECREATEINFO_target_rect',
    'struct__CUVIDGETDECODESTATUS', 'struct__CUVIDH264DPBENTRY',
    'struct__CUVIDH264MVCEXT', 'struct__CUVIDH264PICPARAMS',
    'struct__CUVIDH264SVCEXT', 'struct__CUVIDHEVCPICPARAMS',
    'struct__CUVIDJPEGPICPARAMS', 'struct__CUVIDMPEG2PICPARAMS',
    'struct__CUVIDMPEG4PICPARAMS', 'struct__CUVIDPARSERDISPINFO',
    'struct__CUVIDPARSERPARAMS', 'struct__CUVIDPICPARAMS',
    'struct__CUVIDPROCPARAMS', 'struct__CUVIDRECONFIGUREDECODERINFO',
    'struct__CUVIDRECONFIGUREDECODERINFO_display_area',
    'struct__CUVIDRECONFIGUREDECODERINFO_target_rect',
    'struct__CUVIDSEIMESSAGEINFO', 'struct__CUVIDSOURCEDATAPACKET',
    'struct__CUVIDSOURCEPARAMS', 'struct__CUVIDVC1PICPARAMS',
    'struct__CUVIDVP8PICPARAMS',
    'struct__CUVIDVP8PICPARAMS_0_vp8_frame_tag',
    'struct__CUVIDVP9PICPARAMS', 'struct__CUcontextlock_st',
    'struct__SEIALTERNATIVETRANSFERCHARACTERISTICS',
    'struct__SEICONTENTLIGHTLEVELINFO',
    'struct__SEIMASTERINGDISPLAYINFO', 'struct__TIMECODE',
    'struct__TIMECODEMPEG2', 'struct__TIMECODESET',
    'struct_c__SA_CUAUDIOFORMAT', 'struct_c__SA_CUVIDAV1SEQHDR',
    'struct_c__SA_CUVIDEOFORMAT', 'struct_c__SA_CUVIDEOFORMATEX',
    'struct_c__SA_CUVIDEOFORMAT_display_area',
    'struct_c__SA_CUVIDEOFORMAT_display_aspect_ratio',
    'struct_c__SA_CUVIDEOFORMAT_frame_rate',
    'struct_c__SA_CUVIDEOFORMAT_video_signal_description',
    'struct_c__SA_CUVIDOPERATINGPOINTINFO',
    'struct_c__SA_CUVIDOPERATINGPOINTINFO_0_av1',
    'union__CUVIDH264PICPARAMS_1', 'union__CUVIDH264PICPARAMS_fmo',
    'union__CUVIDPICPARAMS_CodecSpecific',
    'union__CUVIDVP8PICPARAMS_0', 'union_c__SA_CUVIDEOFORMATEX_0',
    'union_c__SA_CUVIDOPERATINGPOINTINFO_0']
