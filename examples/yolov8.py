from extra.models.yolov8 import YOLOv8, box_iou, postprocess, preprocess
import numpy as np
from extra.utils import fetch
import cv2
from collections import defaultdict
import time, sys

def draw_bounding_boxes_and_display(orig_imgs, all_predictions, class_labels, iou_threshold=0.5, is_last=True):
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127
  
  for orig_img, predictions in zip(orig_imgs, all_predictions):
    predictions = np.array(predictions)
    height, width, _ = orig_img.shape
    box_thickness = int((height + width) / 400)
    font_scale = (height + width) / 2500

    grouped_preds = defaultdict(list)
    object_count = defaultdict(int)

    for pred_np in predictions:
      grouped_preds[int(pred_np[-1])].append(pred_np)

    def draw_box_and_label(pred, color):
      x1, y1, x2, y2, conf, _ = pred
      x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
      cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
      label = f"{class_labels[class_id]} {conf:.2f}"
      text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
      label_y, bg_y = (y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1)
      cv2.rectangle(orig_img, (x1, bg_y), (x1 + text_size[0], bg_y + text_size[1]), color, -1)
      font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
      cv2.putText(orig_img, label, (x1, label_y), font, font_scale, font_color, 1, cv2.LINE_AA)

    for class_id, pred_list in grouped_preds.items():
      pred_list = np.array(pred_list)
      while len(pred_list) > 0:
        max_conf_idx = np.argmax(pred_list[:, 4])
        max_conf_pred = pred_list[max_conf_idx]
        pred_list = np.delete(pred_list, max_conf_idx, axis=0)
        color = color_dict[class_labels[class_id]]
        draw_box_and_label(max_conf_pred, color)
        object_count[class_labels[class_id]] += 1
        iou_scores = box_iou(np.array([max_conf_pred[:4]]), pred_list[:, :4])
        low_iou_indices = np.where(iou_scores[0] < iou_threshold)[0]
        pred_list = pred_list[low_iou_indices]
        for low_conf_pred in pred_list:
          draw_box_and_label(low_conf_pred, color)

    print("Objects detected:")
    for obj, count in object_count.items():
      print(f"- {obj}: {count}")
    cv2.imshow('Video', orig_img[...,::-1]) # convert back to BGR
  if is_last:
    print('Press enter to quit')
    cv2.waitKey(0)

def label_predictions(all_predictions):
  class_index_count = defaultdict(int)
  for predictions in all_predictions:
    predictions = np.array(predictions)
    for pred_np in predictions:
      class_id = int(pred_np[-1])
      class_index_count[class_id] += 1
  return dict(class_index_count)

if __name__ == '__main__':
  # usage : python3 yolov8.py "image_URL OR image_path OR 'webcam'" "v8 variant" (optional, n is default)
  # examples :
  # METAL_XCODE=1 python3 examples/yolov8.py docs/logo.png
  # METAL_XCODE=1 python3 examples/yolov8.py webcam n
  # METAL_XCODE=1 python3 examples/yolov8.py https://previews.123rf.com/images/artitwpd/artitwpd1706/artitwpd170600575/80514774-people-walking-riding-bicycle-and-cars-running-on-the-street-in-kolkata-india.jpg s
  # METAL_XCODE=1 python3 examples/yolov8.py test/models/efficientnet/car.jpg m
  # METAL_XCODE=1 python3 examples/yolov8.py test/models/efficientnet/Chicken.jpg l
  # METAL_XCODE=1 python3 examples/yolov8.py ~/Downloads/output.mp4 x
  if len(sys.argv) < 2:
    print("Error: Image URL or path not provided.")
    sys.exit(1)

  img_path = sys.argv[1]
  yolo_variant = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 'n' as the default. Yolov8 has different variants, you can choose from ['n', 's', 'm', 'l', 'x']") or 'n')
  
  print(f'running inference for YOLO version {yolo_variant}')
  yolo_infer = YOLOv8(yolo_variant, num_classes=80)
  yolo_infer.load_from_pretrained()
  
  #v8 and v3 have same 80 class names for Object Detection
  class_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
  class_labels = class_labels.decode('utf-8').split('\n')
  
  if img_path == 'webcam':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
  else:
    cap = cv2.VideoCapture(img_path)
  ret, img = cap.read()
  while cap.isOpened():
    image = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    st = time.time()
    pre_processed_image = preprocess(image)
    print(f'pre processing in {int(round(((time.time() - st) * 1000)))}ms')

    st = time.time()
    predictions = yolo_infer(pre_processed_image)
    print(f'did inference in {int(round(((time.time() - st) * 1000)))}ms')

    st = time.time()
    post_predictions = postprocess(preds=predictions, img=pre_processed_image, orig_imgs=image)
    print(f'post processing in {int(round(((time.time() - st) * 1000)))}ms')

    ret, img = cap.read()
    draw_bounding_boxes_and_display(orig_imgs=image, all_predictions=post_predictions, class_labels=class_labels, is_last=not ret)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if not ret: break
  
  cv2.destroyAllWindows()

# TODO for later:
#  1. AST exp overflow warning while on cpu
#  2. Make NMS faster
