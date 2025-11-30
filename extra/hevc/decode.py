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

  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)
  
  frame_info = frame_info[:4]

  all_slices = []
  # all_datas = []
  # all_bufs_out = []
  # all_pics = []
  with Timing("prep slices to gpu: "):
    opaque_nv = opaque.to("NV").contiguous().realize()
    
    # raw image out from decoder
    # bufout = Tensor.empty(len(frame_info), out_image_size, dtype=dtypes.uint8).realize()
    # pics = Tensor.empty(len(frame_info), h, w, 3, dtype=dtypes.uint8).realize()

    for i, (offset, sz, frame_pos, history_sz, _) in enumerate(frame_info):
      all_slices.append(hevc_tensor[offset:offset+sz].to("NV").contiguous().realize())

      # bufout = Tensor.empty(out_image_size, dtype=dtypes.uint8).realize()
      # bufout.uop.buffer.allocate()
      # all_bufs_out.append(bufout)

      # if getenv("VALIDATE", 0):
      #   pic = Tensor.empty((h + (h + 1) // 2) * w, dtype=dtypes.uint8).realize()
      # else:
      #   pic = Tensor.empty(h, w, 3, dtype=dtypes.uint8).realize()
      # pic.uop.buffer.allocate()
      # all_pics.append(pic)
    Device.default.synchronize()

  @TinyJit
  def decode_jit(src:Tensor, data:Tensor, pos:Variable, *hist:Tensor):
    return src.decode_hevc_frame(pos, out_image_size, data, hist).realize()

  @TinyJit
  def untile_nv12(src:Tensor):
    luma = src.reshape(-1)[_addr_table(h, w, round_up(luma_w, 64))]
    chroma = src.reshape(-1)[chroma_off:][_addr_table((h + 1) // 2, w, round_up(luma_w, 64))]
    # return nv12_to_bgr_from_planes(luma, chroma, h, w)
    return luma.cat(chroma).realize()

  max_hist = max(history_sz for _, _, _, history_sz, _ in frame_info)
  history = [Tensor.empty(out_image_size, dtype=dtypes.uint8).realize() for x in range(max_hist)]

  pos = Variable("pos", 0, max_hist + 1)

  # warm up
  # for i in range(3):
  #   x = decode_jit(all_slices[0], opaque_nv[0], pos.bind(frame_pos), *history)
  #   x = untile_nv12(x)

  out_images = []
  
  with Timing("decoding whole file: ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
    for i, (offset, sz, frame_pos, history_sz, is_hist) in enumerate(frame_info):
      history = history[-max_hist:]

      # print(id(history[0]))
      x = all_slices[i].decode_hevc_frame(pos.bind(frame_pos), out_image_size, opaque_nv[i], history).realize()
      # x = decode_jit(all_slices[i], opaque_nv[i], pos.bind(frame_pos), *history)

      # if getenv("VALIDATE", 0):
      #   img = untile_nv12(x)

      out_images.append(untile_nv12(x).clone().realize())
      if is_hist: history.append(x.clone().realize())
      Device.default.synchronize()

  import cv2
  if getenv("VALIDATE", 0):
    import pickle
    decoded_frames = pickle.load(open("extra/hevc/cuda_decoded_frames.pkl", "rb"))

  for i, img in tqdm(enumerate(out_images)):
    if getenv("VALIDATE", 0):
      if i < len(decoded_frames):
        assert img.data() == decoded_frames[i], f"Frame {i} does not match reference decoder!"
    else:
      cv2.imwrite(f"{args.output_dir}/zcm_frame_{i:04d}.png", img.numpy())
