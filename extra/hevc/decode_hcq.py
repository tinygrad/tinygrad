import cv2, argparse, os
from tinygrad.runtime.ops_nv import NVVideoQueue
from tinygrad.helpers import getenv, DEBUG
from tinygrad.helpers import DEBUG, round_up, ceildiv, Timing, prod, tqdm
from extra.hevc.hevc import parse_hevc_file_headers, _addr_table, nv12_to_bgr_from_planes, nv_gpu
from tinygrad import Tensor, dtypes, Device, TinyJit

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
    w, h, luma_w, luma_h, frame_info = parse_hevc_file_headers(dat)

  offset, sz, shape, opaque, frame_pos, history_sz, _ = frame_info[0]
  chroma_off = round_up(shape[0], 64) * round_up(shape[1], 64)

  all_slices = []
  all_datas = []
  all_bufs_out = []
  all_pics = []
  with Timing("prep slices to gpu: "):
    for i, (offset, sz, shape, opaque, frame_pos, history_sz, _) in enumerate(frame_info):
      all_slices.append(dat_nv[offset:offset+sz].contiguous().realize())
      all_datas.append(opaque.contiguous().realize())

      shape = (round_up(shape[0] + (shape[0] + 1) // 2, 64), round_up(shape[1], 64))
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
  def untile_nv12(src:Tensor, out:Tensor):
    luma = src.reshape(-1)[_addr_table(h, w, round_up(w, 64))]
    chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(w, 64))]
    x = luma.cat(chroma) if getenv("VALIDATE", 0) else nv12_to_bgr_from_planes(luma, chroma, h, w)
    out.assign(x).realize()
    return x.realize()

  # warm up
  for i in range(3): x = untile_nv12(all_bufs_out[0], all_pics[0])

  dev = Device.default
  dev._ensure_has_vid_hw(luma_w, luma_h)

  history = []
  prev_timeline_wait = dev.timeline_value - 1
  with Timing("decoding whole file (hcq): ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
    for i, (offset, sz, shape, opaque, frame_pos, history_sz, is_hist) in enumerate(frame_info):
      history = history[-history_sz:] if history_sz > 0 else []

      bufin_hcq = all_slices[i].uop.buffer._buf
      desc_buf_hcq = all_datas[i].uop.buffer._buf
      bufout_hcq = all_bufs_out[i].uop.buffer._buf

      if DEBUG >= 6: dump_struct_ext(nv_gpu.nvdec_hevc_pic_s.from_buffer(opaque.data()))

      NVVideoQueue().wait(dev.timeline_signal, prev_timeline_wait) \
                    .decode_hevc_chunk(i, desc_buf_hcq, bufin_hcq, bufout_hcq, frame_pos, history,
                                       [(frame_pos-x) % (len(history) + 1) for x in range(len(history), 0, -1)],
                                       chroma_off, dev.vid_coloc_buf, dev.vid_filter_buf, dev.intra_top_off, dev.vid_status_buf) \
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
