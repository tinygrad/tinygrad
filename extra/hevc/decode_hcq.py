import cv2, argparse, os
from tinygrad.runtime.ops_nv import NVVideoQueue
from tinygrad.helpers import getenv, DEBUG
from tinygrad.helpers import DEBUG, round_up, ceildiv, Timing, prod, tqdm
from extra.hevc.hevc import parse_hevc_file_headers, _addr_table, nv12_to_bgr_from_planes, nv_gpu
from tinygrad import Tensor, dtypes, Device, TinyJit, Variable

if DEBUG >= 6:
  from extra.nv_gpu_driver.nv_ioctl import dump_struct_ext

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str, default="")
  parser.add_argument("--output_dir", type=str, default="extra/hevc/out")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  if args.input_file == "":
    url = "https://github.com/commaai/comma2k19/raw/refs/heads/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc"
    hevc_tensor = Tensor.from_url(url, device="CPU")
  else:
    hevc_tensor = Tensor.empty(os.stat(args.input_file).st_size, dtype=dtypes.uint8, device=f"disk:{args.input_file}").to("CPU")

  with Timing("prep infos: "):
    dat = bytes(hevc_tensor.data())
    dat_nv = hevc_tensor.to("NV")
    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat)

  # offset, sz, _, _, frame_pos, history_sz, _ = frame_info[0]
  # chroma_off = round_up(shape[0], 64) * round_up(shape[1], 64)

  frame_info = frame_info[:100]

  all_slices = []
  # all_datas = []
  all_bufs_out = []
  all_pics = []
  with Timing("prep slices to gpu: "):
    opaque_nv = opaque.to("NV").realize()
    
    # raw image out from decoder
    # bufout = Tensor.empty(len(frame_info), luma_h + (luma_h + 1) // 2, round_up(luma_w, 64), dtype=dtypes.uint8).realize()
    # pics = Tensor.empty(len(frame_info), *((h + (h + 1) // 2, ) * w if getenv("VALIDATE", 0) else (h, w, 3)), dtype=dtypes.uint8).realize()

    for i, (offset, sz, frame_pos, history_sz, _) in enumerate(frame_info):
      all_slices.append(dat_nv[offset:offset+sz].contiguous().realize())

      shape = (luma_h + (luma_h + 1) // 2, round_up(luma_w, 64))
      bufout = Tensor.empty(shape, dtype=dtypes.uint8).realize()
      bufout.uop.buffer.allocate()
      all_bufs_out.append(bufout)

      if getenv("VALIDATE", 0):
        pic = Tensor.empty((h + (h + 1) // 2) * w, dtype=dtypes.uint8).realize()
      else:
        pic = Tensor.empty(h, w, 3, dtype=dtypes.uint8).realize()
      pic.uop.buffer.allocate()
      all_pics.append(pic)
    Device.default.synchronize()

  @TinyJit
  def decode_jit(src:Tensor, hist:Tensor, data:Tensor, out:Tensor, pos:Variable):
    x = src.decode_hevc_frame(pos, (luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)), data, [hist]).realize()
    out.assign(x).realize()
    return x

  @TinyJit
  def untile_nv12(src:Tensor, out:Tensor):
    luma = src.reshape(-1)[_addr_table(h, w, round_up(luma_w, 64))]
    chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(luma_w, 64))]
    x = luma.cat(chroma) if getenv("VALIDATE", 0) else nv12_to_bgr_from_planes(luma, chroma, h, w)
    out.assign(x).realize()
    return x.realize()

  # warm up
  for i in range(3): x = untile_nv12(all_bufs_out[0], all_pics[0])

  # max_hist = 1
  # history = [Tensor.empty(luma_h + (luma_h + 1) // 2, round_up(luma_w, 64), dtype=dtypes.uint8).realize() for x in range(max_hist)]
  # for i in range(3): x = decode_jit(all_slices[0], history[0], all_datas[0], all_bufs_out[0], Variable("pos", 0, 6).bind(frame_pos))

  # var = Variable("pos", 0, 6)
  # with Timing("decoding whole file: ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
  #   for i, (offset, sz, shape, opaque, frame_pos, history_sz, is_hist) in enumerate(frame_info):
  #     history = history[-max_hist:] # if history_sz > 0 else []

  #     # pos = var.bind(frame_pos)
  #     x = decode_jit(all_slices[i], history[0], all_datas[i], all_bufs_out[i], var.bind(frame_pos))
  #     if is_hist: history.append(all_bufs_out[i])
  #   Device.default.synchronize()

  # # res = untile_nv12(history[-1])
  # import cv2
  # cv2.imwrite(f"{args.output_dir}/zcm_frame_{i:04d}.png", all_pics[-1].numpy())

  # exit(0)

  dev = Device.default
  dev._ensure_has_vid_hw(luma_w, luma_h)

  history = []
  prev_timeline_wait = dev.timeline_value - 1
  with Timing("decoding whole file (hcq): ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
    for i, (_, _, frame_pos, history_sz, is_hist) in enumerate(frame_info):
      history = history[-history_sz:] if history_sz > 0 else []

      bufin_hcq = all_slices[i].uop.buffer._buf
      desc_buf_hcq = opaque_nv.uop.buffer._buf.offset(i * prod(opaque_nv.shape[1:]), prod(opaque_nv.shape[1:]))
      # bufout_hcq = bufout.uop.buffer._buf.offset(i * prod(bufout.shape[1:]), prod(bufout.shape[1:]))
      bufout_hcq = all_bufs_out[i].uop.buffer._buf

      if DEBUG >= 6: dump_struct_ext(nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque.data()))

      NVVideoQueue().wait(dev.timeline_signal, prev_timeline_wait) \
                    .decode_hevc_chunk(i, desc_buf_hcq, bufin_hcq, bufout_hcq, frame_pos, history,
                                       [(frame_pos-x) % (len(history) + 1) for x in range(len(history), 0, -1)],
                                       chroma_off, dev.vid_coloc_buf, dev.vid_filter_buf, dev.intra_top_off, dev.vid_stat_buf) \
                    .signal(dev.timeline_signal, prev_timeline_wait:=dev.next_timeline()).submit(dev)
      untile_nv12(all_bufs_out[i], all_pics[i])
      if is_hist: history.append(bufout_hcq)
    Device.default.synchronize()

  import cv2
  if getenv("VALIDATE", 0):
    import pickle
    decoded_frames = pickle.load(open("extra/hevc/cuda_decoded_frames.pkl", "rb"))

  for i, src in tqdm(enumerate(all_pics)):
    if getenv("VALIDATE", 0):
      if i < len(decoded_frames):
        assert all_pics[i].data() == decoded_frames[i], f"Frame {i} does not match reference decoder!"
    else:
      cv2.imwrite(f"{args.output_dir}/cm_frame_{i:04d}.png", all_pics[i].numpy())
