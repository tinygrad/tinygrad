from models.mask_rcnn import MaskRCNN
from models.resnet import ResNet
from models.mask_rcnn import BoxList, F
from torchvision import transforms as T
from tinygrad.tensor import Tensor
from PIL import Image
import numpy as np
import torch
import argparse
import cv2

to_bgr_transform = T.Lambda(lambda x: x * 255)

normalize_transform = T.Normalize(
  mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]
)

transforms = T.Compose(
  [
    T.ToPILImage(),
    T.Resize(size=(1024, 1024)),
    T.ToTensor(),
    to_bgr_transform,
    normalize_transform,
  ]
)


def expand_boxes(boxes, scale):
  w_half = (boxes[:, 2] - boxes[:, 0]) * .5
  h_half = (boxes[:, 3] - boxes[:, 1]) * .5
  x_c = (boxes[:, 2] + boxes[:, 0]) * .5
  y_c = (boxes[:, 3] + boxes[:, 1]) * .5

  w_half *= scale
  h_half *= scale

  boxes_exp = torch.zeros_like(boxes)
  boxes_exp[:, 0] = x_c - w_half
  boxes_exp[:, 2] = x_c + w_half
  boxes_exp[:, 1] = y_c - h_half
  boxes_exp[:, 3] = y_c + h_half
  return boxes_exp


def expand_masks(mask, padding):
  N = mask.shape[0]
  M = mask.shape[-1]
  pad2 = 2 * padding
  scale = float(M + pad2) / M
  padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
  padded_mask[:, :, padding:-padding, padding:-padding] = mask
  return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
  mask = torch.tensor(mask.numpy())
  box = torch.tensor(box.numpy())
  padded_mask, scale = expand_masks(mask[None], padding=padding)
  mask = padded_mask[0, 0]
  box = expand_boxes(box[None], scale)[0]
  box = box.to(dtype=torch.int32)

  TO_REMOVE = 1
  w = int(box[2] - box[0] + TO_REMOVE)
  h = int(box[3] - box[1] + TO_REMOVE)
  w = max(w, 1)
  h = max(h, 1)

  # Set shape to [batchxCxHxW]
  mask = mask.expand((1, 1, -1, -1))

  # Resize mask
  mask = mask.to(torch.float32)
  mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
  mask = mask[0][0]

  if thresh >= 0:
    mask = mask > thresh
  else:
    # for visualization and debugging, we also
    # allow it to return an unmodified mask
    mask = (mask * 255).to(torch.uint8)

  im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
  x_0 = max(box[0], 0)
  x_1 = min(box[2] + 1, im_w)
  y_0 = max(box[1], 0)
  y_1 = min(box[3] + 1, im_h)

  im_mask[y_0:y_1, x_0:x_1] = mask[
                              (y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])
                              ]
  return im_mask


class Masker(object):
  """
  Projects a set of masks in an image on the locations
  specified by the bounding boxes
  """

  def __init__(self, threshold=0.5, padding=1):
    self.threshold = threshold
    self.padding = padding

  def forward_single_image(self, masks, boxes):
    boxes = boxes.convert("xyxy")
    im_w, im_h = boxes.size
    res = [
      paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
      for mask, box in zip(masks, boxes.bbox)
    ]
    if len(res) > 0:
      res = torch.stack(res, dim=0)[:, None]
    else:
      res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
    return Tensor(res.numpy())

  def __call__(self, masks, boxes):
    if isinstance(boxes, BoxList):
      boxes = [boxes]

    # Make some sanity check
    assert len(boxes) == len(masks), "Masks and boxes should have the same length."

    results = []
    for mask, box in zip(masks, boxes):
      assert mask.shape[0] == len(box), "Number of objects should be the same."
      result = self.forward_single_image(mask, box)
      results.append(result)
    return results


masker = Masker(threshold=0.5, padding=1)

def compute_prediction(original_image, model_type='tiny'):
  # apply pre-processing to image
  image = transforms(np.asarray(original_image))
  print(image.shape)
  image = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2]).numpy()
  image = Tensor(image, requires_grad=False)

  predictions = model_tiny(image)

  # always single image is passed at a time
  prediction = predictions[0]

  # reshape prediction (a BoxList) into the original image size
  height, width = original_image.shape[:-1]
  prediction = prediction.resize((width, height))

  if prediction.has_field("mask"):
    # if we have masks, paste the masks in the right position
    # in the image, as defined by the bounding boxes
    masks = prediction.get_field("mask")
    # always single image is passed at a time
    masks = masker([masks], [prediction])[0]
    if model_type != 'tiny':
      masks = torch.tensor(masks.numpy())
    prediction.add_field("mask", masks)
  return prediction

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    l = torch.tensor(labels[:, None].numpy())
    colors = l * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    image = np.asarray(image)
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite

CATEGORIES = [
        "__background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = torch.tensor(predictions.get_field("labels").numpy())
    boxes = predictions.bbox
    image = np.asarray(image)
    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = torch.tensor(box.numpy())
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").numpy().tolist()
    labels = predictions.get_field("labels").numpy().tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox.numpy()
    image = np.asarray(image)
    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        x, y = int(x), int(y)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def select_top_predictions(predictions, confidence_threshold=0.9):
  scores = torch.tensor(predictions.get_field("scores").numpy())
  keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
  predictions = predictions[keep]
  scores = torch.tensor(predictions.get_field("scores").numpy())
  _, idx = scores.sort(0, descending=True)
  return predictions[idx]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run MaskRCNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image', type=str, help="Path of the image to run")
  parser.add_argument('--threshold', type=float, default=0.7, help="Detector threshold")
  parser.add_argument('--out', type=str, default="/tmp/rendered.png", help="Output filename")
  args = parser.parse_args()

  resnet = ResNet(50, num_classes=None)
  model_tiny = MaskRCNN(resnet)
  model_tiny.load_from_pretrained()
  img = cv2.imread(args.image, cv2.IMREAD_COLOR)
  img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  result = compute_prediction(img)
  top_result_tiny = select_top_predictions(result, confidence_threshold=args.threshold)
  bbox_image = overlay_boxes(img_rbg, top_result_tiny)
  mask_image = overlay_mask(bbox_image, top_result_tiny)
  final_image = overlay_class_names(mask_image, top_result_tiny)

  im = Image.fromarray(final_image)
  print(f"saving {args.out}")
  im.save(args.out)
  im.show()
