import numpy as np
from typing import List, Tuple

# Changes the bounding box coordinates in accordance with the new image size.
def resize_box_based_on_new_image_size(box: List[float], img_old_size: Tuple[int], img_new_size: Tuple[int]) -> List[float]:
  ratio_height, ratio_width = [new / orig for new, orig in zip(img_new_size[:2], img_old_size[:2])]
  xmin, ymin, xmax, ymax = box
  xmin *= ratio_width
  xmax *= ratio_width
  ymin *= ratio_height
  ymax *= ratio_height
  return [xmin, ymin, xmax, ymax]

def bbox_transform(proposals, reference_boxes):
  """
  Encodes the proposals into the representation used for training the regressor
  by calculating relative spatial transformations from the proposals to the reference boxes.
  Returns the encoded targets as an array.
  """

  proposals_x1 = proposals[:, 0].reshape(-1, 1)
  proposals_y1 = proposals[:, 1].reshape(-1, 1)
  proposals_x2 = proposals[:, 2].reshape(-1, 1)
  proposals_y2 = proposals[:, 3].reshape(-1, 1)

  reference_boxes_x1 = reference_boxes[:, 0].reshape(-1, 1)
  reference_boxes_y1 = reference_boxes[:, 1].reshape(-1, 1)
  reference_boxes_x2 = reference_boxes[:, 2].reshape(-1, 1)
  reference_boxes_y2 = reference_boxes[:, 3].reshape(-1, 1)

  # implementation starts here
  ex_widths = proposals_x2 - proposals_x1
  ex_heights = proposals_y2 - proposals_y1
  ex_ctr_x = proposals_x1 + 0.5 * ex_widths
  ex_ctr_y = proposals_y1 + 0.5 * ex_heights

  gt_widths = reference_boxes_x2 - reference_boxes_x1
  gt_heights = reference_boxes_y2 - reference_boxes_y1
  gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
  gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.concatenate((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
  return targets
