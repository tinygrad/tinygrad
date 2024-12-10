import argparse, shutil, os, pathlib, json, urllib
from multiprocessing import Pool
import pandas as pd # type: ignore
from tinygrad.helpers import tqdm

BASEDIR = pathlib.Path(__file__).parent/"COCO"
BASEDIR.mkdir(exist_ok=True)

def download_coco2014_5k(extra_args):
  # Adapted from: https://github.com/mlcommons/inference/blob/master/text_to_image/tools/coco.py
  import subprocess
  import zipfile

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--download-images', action='store_true')
  parser.add_argument('-v', '--verify-images', action='store_true')
  parser.add_argument('-n', '--num-workers', type=int, default=1)
  args = parser.parse_args(args=extra_args)

  MAX_IMAGES = 5000
  SEED = 2023
  rootdir = BASEDIR/'coco2014_5k'
  rootdir.mkdir(exist_ok=True)

  output_filepath = rootdir/"captions.tsv"
  annotion_filepath = rootdir/"annotations_trainval2014.zip"
  rawpath = rootdir/"raw/"

  if output_filepath.exists():
    print(f"Already found output file: {output_filepath}")
  else:
    if not annotion_filepath.exists():
      subprocess.run(["wget", "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"], cwd=str(rootdir))

    if not rawpath.exists():
      with zipfile.ZipFile(annotion_filepath, "r") as zip_ref:
        zip_ref.extractall(str(rawpath))

    captions_filepath = rawpath/"annotations/captions_val2014.json"
    with open(captions_filepath, "r") as f:
      data = json.load(f)
      annotations = data["annotations"]
      images      = data["images"]
    df_annotations = pd.DataFrame(annotations)
    df_images = pd.DataFrame(images)
    df_annotations = df_annotations.drop_duplicates(subset=["image_id"], keep="first")
    # Sort, shuffle and choose the final dataset
    df_annotations = df_annotations.sort_values(by=["id"])
    df_annotations = df_annotations.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df_annotations = df_annotations.iloc[:MAX_IMAGES]
    df_annotations["caption"] = df_annotations["caption"].apply(lambda x: x.replace("\n", "").strip())
    df_annotations = (
      df_annotations.merge(df_images, how="inner", left_on="image_id", right_on="id")
                    .drop(["id_y"], axis=1)
                    .rename(columns={"id_x": "id"})
                    .sort_values(by=["id"])
                    .reset_index(drop=True)
    )
    df_annotations[
        ["id", "image_id", "caption", "height", "width", "file_name", "coco_url"]
    ].to_csv(str(output_filepath), sep="\t", index=False)

  if annotion_filepath.exists():
    os.remove(str(annotion_filepath))
  if rawpath.exists():
    shutil.rmtree(rawpath)

  if args.download_images:
    calibration_dirpath = rootdir/"calibration/"
    calibration_dirpath.mkdir(exist_ok=True)
    df_annotations = pd.read_csv(str(output_filepath), sep="\t")
    tasks = [(row["coco_url"], str(calibration_dirpath), row["file_name"]) for _,row in df_annotations.iterrows()]
    def download_img(args):
      img_url, target_folder, file_name = args
      dest_path = f"{target_folder}/{file_name}"
      if not os.path.exists(dest_path):
        urllib.request.urlretrieve(img_url, dest_path)
    for task in tqdm(tasks):
      download_img(task)

  if args.verify_images:
    from tinygrad import Tensor
    from PIL import Image
    import numpy as np
    df_annotations = pd.read_csv(str(output_filepath), sep="\t")
    calibration_dirpath = rootdir/"calibration/"
    if not calibration_dirpath.exists():
      print("ERROR: calibration_dirpath did not exist, make sure to --download-images")
    else:
      print("Verifying images...")
      total, ok_count = 0, 0
      for filename in tqdm(df_annotations["file_name"]):
        total += 1
        image_filepath = calibration_dirpath/filename
        if not image_filepath.exists():
          print(f"ERROR: missing image {filename}")
        else:
          try:
            Tensor(np.array(Image.open(image_filepath))).realize()
            ok_count += 1
          except Exception as ex:
            print(f"ERROR: Got exception loading {filename}, {ex}")
    print(f"Verified {ok_count}/{total} images downloaded ok")

  latents_filepath = rootdir/"latents.npy"
  if not latents_filepath.exists():
    subprocess.run(["wget", "https://github.com/mlcommons/inference/raw/55bebbfcb2a82ee64048674a2d9862ee0ab672eb/text_to_image/tools/latents.npy"], cwd=str(latents_filepath.parent))

if __name__ == "__main__":
  dataset_map = {
    "coco2014_5k": download_coco2014_5k,
  }
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", choices=list(dataset_map.keys()))
  args, unkown = parser.parse_known_args()
  dataset_map[args.dataset](unkown)
