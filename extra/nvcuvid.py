import ctypes, queue
from tinygrad.runtime.ops_cuda import CUDADevice
from tinygrad.runtime.autogen import cuda, nvcuvid
from tinygrad.helpers import DEBUG, init_c_var

from tinygrad.ops import UOp
from tinygrad import dtypes
from tinygrad.tensor import Tensor

def check(status):
  if status != 0: raise RuntimeError(f"NVCUVID Error {status}")

class CUDAVideoDecoder:
  def __init__(self, dev:CUDADevice, codec=nvcuvid.cudaVideoCodec_HEVC) -> None:
    self.dev = dev
    cu_ctx = ctypes.cast(dev.context, nvcuvid.CUcontext)
    self.lock = init_c_var(nvcuvid.CUvideoctxlock(), lambda x: check(nvcuvid.cuvidCtxLockCreate(ctypes.byref(x), cu_ctx)))
    self.stream = init_c_var(cuda.CUstream(), lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_DEFAULT)))
    self._userdata = ctypes.py_object(self) # keep a ref
    params = nvcuvid.CUVIDPARSERPARAMS()
    params.CodecType = codec
    params.pUserData = ctypes.cast(ctypes.pointer(self._userdata), ctypes.c_void_p)
    params.pfnSequenceCallback = CUDAVideoDecoder._sequence_callback
    params.pfnDecodePicture = CUDAVideoDecoder._decode_callback
    params.pfnDisplayPicture = CUDAVideoDecoder._display_callback
    self.parser = init_c_var(nvcuvid.CUvideoparser(), lambda x: check(nvcuvid.cuvidCreateVideoParser(ctypes.byref(x), ctypes.byref(params))))
    self.decoder, self.decode_pic_counter, self.pic_num_in_decode_order, self.frame_queue = None, 0, {}, queue.Queue()

  def decode(self, data: memoryview) -> int:
    self._decoded_frame = self._decoded_frame_returned = 0
    packet = nvcuvid.CUVIDSOURCEDATAPACKET()
    payload_buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    packet.payload_size = len(data) 
    packet.payload = ctypes.cast(payload_buffer, ctypes.POINTER(ctypes.c_ubyte))
    check(nvcuvid.cuvidParseVideoData(self.parser, ctypes.byref(packet)))
    return self._decoded_frame
    
  @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(nvcuvid.CUVIDEOFORMAT))
  def _sequence_callback(userdata, videoformat) -> int:
    inst = ctypes.cast(userdata, ctypes.POINTER(ctypes.py_object)).contents.value
    if DEBUG >= 4:
      print("codec:", nvcuvid.cudaVideoCodec__enumvalues[videoformat.contents.codec])
      print("frame rate:", videoformat.contents.frame_rate.numerator, "/", videoformat.contents.frame_rate.denominator)
      print("sequence:", "progressive" if videoformat.contents.progressive_sequence else "interlaced")
      print("coded size:", videoformat.contents.coded_width, "x", videoformat.contents.coded_height)
      print("display area:", videoformat.contents.display_area.left, videoformat.contents.display_area.top, videoformat.contents.display_area.right, videoformat.contents.display_area.bottom)
      print("chroma format:", nvcuvid.cudaVideoChromaFormat__enumvalues[videoformat.contents.chroma_format])
      print("bit depth:", videoformat.contents.bit_depth_luma_minus8 + 8)

    caps = nvcuvid.CUVIDDECODECAPS()
    caps.eCodecType = videoformat.contents.codec
    caps.eChromaFormat = videoformat.contents.chroma_format
    caps.nBitDepthMinus8 = videoformat.contents.bit_depth_luma_minus8
    check(cuda.cuCtxSetCurrent(inst.dev.context))
    check(nvcuvid.cuvidGetDecoderCaps(ctypes.byref(caps)))
    if not caps.bIsSupported: raise RuntimeError("codec not supported")
    if videoformat.contents.coded_width > caps.nMaxWidth or videoformat.contents.coded_height > caps.nMaxHeight:
      raise RuntimeError(f"resolution {videoformat.contents.coded_width}x{videoformat.contents.coded_height} exceeds max {caps.nMaxWidth}x{caps.nMaxHeight} on this device")

    if inst.decoder: check(nvcuvid.cuvidDestroyDecoder(inst.decoder))
    create_info = nvcuvid.CUVIDDECODECREATEINFO()
    create_info.CodecType = videoformat.contents.codec
    create_info.ChromaFormat = videoformat.contents.chroma_format
    create_info.OutputFormat = nvcuvid.cudaVideoSurfaceFormat_NV12
    create_info.bitDepthMinus8 = videoformat.contents.bit_depth_luma_minus8
    create_info.DeinterlaceMode = nvcuvid.cudaVideoDeinterlaceMode_Weave if videoformat.contents.progressive_sequence else nvcuvid.cudaVideoDeinterlaceMode_Adaptive
    create_info.ulNumDecodeSurfaces = videoformat.contents.min_num_decode_surfaces
    create_info.ulNumOutputSurfaces = 2
    create_info.ulCreationFlags = nvcuvid.cudaVideoCreate_PreferCUVID
    create_info.vidLock = inst.lock
    create_info.ulWidth = videoformat.contents.coded_width
    create_info.ulHeight = videoformat.contents.coded_height
    create_info.ulTargetWidth = videoformat.contents.coded_width
    create_info.ulTargetHeight = videoformat.contents.coded_height
    check(cuda.cuCtxSetCurrent(inst.dev.context))
    inst.decoder = init_c_var(nvcuvid.CUvideodecoder(), lambda x: check(nvcuvid.cuvidCreateDecoder(ctypes.byref(x), ctypes.byref(create_info))))
    inst.frame_width = (videoformat.contents.display_area.right - videoformat.contents.display_area.left)
    inst.luma_height = (videoformat.contents.display_area.bottom - videoformat.contents.display_area.top)
    inst.chroma_height = (inst.luma_height + 1) // 2 # nv12
    inst.surface_height = videoformat.contents.coded_height
    return videoformat.contents.min_num_decode_surfaces

  @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(nvcuvid.CUVIDPICPARAMS))
  def _decode_callback(userdata, pic_params) -> int:
    inst = ctypes.cast(userdata, ctypes.POINTER(ctypes.py_object)).contents.value
    assert inst.decoder, "decoder not initialized"

    curr_pic_index = pic_params.contents.CurrPicIdx
    inst.pic_num_in_decode_order[curr_pic_index] = inst.decode_pic_counter
    inst.decode_pic_counter += 1

    check(cuda.cuCtxSetCurrent(inst.dev.context))
    check(nvcuvid.cuvidDecodePicture(inst.decoder, pic_params))
    return 1

  @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(nvcuvid.CUVIDPARSERDISPINFO))
  def _display_callback(userdata, disp_info) -> int:
      inst = ctypes.cast(userdata, ctypes.POINTER(ctypes.py_object)).contents.value
      d = disp_info.contents

      proc = nvcuvid.CUVIDPROCPARAMS()
      proc.progressive_frame = d.progressive_frame
      proc.second_field = d.repeat_first_field + 1
      proc.top_field_first = d.top_field_first
      proc.unpaired_field = (d.repeat_first_field < 0)
      proc.output_stream = ctypes.cast(inst.stream, ctypes.POINTER(nvcuvid.struct_CUstream_st))
      
      src_frame, src_pitch = ctypes.c_uint64(0), ctypes.c_uint32(0)
      check(cuda.cuCtxSetCurrent(inst.dev.context))
      check(nvcuvid.cuvidMapVideoFrame64(inst.decoder, d.picture_index, ctypes.byref(src_frame), ctypes.byref(src_pitch), ctypes.byref(proc)))

      status = nvcuvid.CUVIDGETDECODESTATUS()
      check(nvcuvid.cuvidGetDecodeStatus(inst.decoder, d.picture_index, ctypes.byref(status)))
      if status.decodeStatus in (nvcuvid.cuvidDecodeStatus_Error, nvcuvid.cuvidDecodeStatus_Error_Concealed):
          print("decode error occurred for picture", inst.pic_num_in_decode_order.get(d.picture_index, -1), file=sys.stderr)

      fw, lh, ch = inst.frame_width, inst.luma_height, inst.chroma_height
      host_buffer = (ctypes.c_ubyte * fw * (lh + ch))()

      chroma_offset = src_pitch.value * (((inst.surface_height + 1) & ~1))
      copy_luma = cuda.CUDA_MEMCPY2D(
          srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src_frame.value, srcPitch=src_pitch.value,
          dstMemoryType=cuda.CU_MEMORYTYPE_HOST,   dstHost=ctypes.cast(host_buffer, ctypes.c_void_p),
          dstPitch=fw, WidthInBytes=fw, Height=lh
      )
      copy_chroma = cuda.CUDA_MEMCPY2D(
          srcDevice=src_frame.value + chroma_offset, srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE,
          srcPitch=src_pitch.value, dstY=lh, dstMemoryType=cuda.CU_MEMORYTYPE_HOST,
          dstHost=ctypes.cast(host_buffer, ctypes.c_void_p), dstPitch=fw,
          WidthInBytes=fw, Height=ch
      )

      check(cuda.cuMemcpy2DAsync_v2(ctypes.byref(copy_luma), inst.stream))
      check(cuda.cuMemcpy2DAsync_v2(ctypes.byref(copy_chroma), inst.stream))
      check(cuda.cuStreamSynchronize(inst.stream))
      check(nvcuvid.cuvidUnmapVideoFrame64(inst.decoder, src_frame.value))
      
      inst.frame_queue.put(bytes(host_buffer))
      inst._decoded_frame += 1
      return 1

if __name__ == "__main__":
  import cv2, numpy as np
  from tinygrad.device import Device
  from pathlib import Path

  def show_frames():
    while not dec.frame_queue.empty():
      w, h, ch = dec.frame_width, dec.luma_height, dec.chroma_height
      total = h + ch
      frame = np.frombuffer(dec.frame_queue.get(), dtype=np.uint8).reshape((total, w))
      cv2.imshow("Decoded Frame", cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
      cv2.waitKey(42)

  dec = CUDAVideoDecoder(Device.default)
  with (Path(__file__).parent / "big_buck_bunny.hevc").open("rb") as f:
    for packet in iter(lambda: f.read(4096), b""):
      dec.decode(packet)
      show_frames()
  show_frames()
  cv2.destroyAllWindows()