from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from typing import Any
from models.retinanet import RetinaNet
from models.resnet import ResNeXt50_32X4D
from extra.datasets import openimages
from PIL import Image
from extra.datasets.openimages import openimages, iterate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tqdm import trange
import numpy as np
import torch
from examples.hlb_cifar10 import cross_entropy
from typing import List, Tuple
from extra.training import focal_loss, smooth_l1_loss
NUM = getenv("NUM", 2)
BS = getenv("BS", 8)
CNT = getenv("CNT", 10)
BACKWARD = getenv("BACKWARD", 0)
TRAINING = getenv("TRAINING", 1)
ADAM = getenv("ADAM", 0)
CLCACHE = getenv("CLCACHE", 0)
IMAGE_SIZES = {"debug" : (200,200), "mlperf" : (800,800)}

def resize_box_based_on_new_image_size(box: List[float], img_old_size: Tuple[int], img_new_size: Tuple[int]) -> List[float]:
  ratio_height, ratio_width = [new / orig for new, orig in zip(img_new_size[:2], img_old_size[:2])]
  xmin, ymin, xmax, ymax = box
  xmin *= ratio_width
  xmax *= ratio_width
  ymin *= ratio_height
  ymax *= ratio_height
  return [xmin, ymin, xmax, ymax]

def iou(anchor, ann_box):
    #TODO tensorize this function for lazyness exploitation
    intersection_x = max(0,min(anchor[2],ann_box[2])-max(anchor[0],ann_box[0]))
    intersection_y = max(0,min(anchor[3],ann_box[3])-max(anchor[1],ann_box[1]))
    intersection = intersection_x * intersection_y
    anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    ann_box_area = (ann_box[2] - ann_box[0]) * (ann_box[3] - ann_box[1])
    union = anchor_area + ann_box_area - intersection
    return intersection / union

def foreground_background_matrix(anchors, ann_boxes):
    #TODO 1 tensorize this function for lazyness exploitation
    #TODO 2 (alternative to 1) this matrix is most certainly sparse... edge list? adjacency list? dict of dicts?
    overlaps = np.zeros((len(anchors), len(ann_boxes)))
    for i in range(len(anchors)):
        for j in range(len(ann_boxes)):
            pair_iou = iou(anchors[i], ann_boxes[j])
            if pair_iou >= 0.5:
                overlaps[i,j] = 1
            elif pair_iou<0.4:
                overlaps[i,j] = -1 #background
    return overlaps
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

class RetinaNetTrainer:
    def __init__(self, model : RetinaNet = RetinaNet(ResNeXt50_32X4D())):
        self.model = model
        self.optimizer = optim.SGD(get_parameters(model), lr=0.001)   
        self.input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.dataset = COCO(openimages())
        self.image_size = IMAGE_SIZES["debug"]
    def get_ground_truths(self, anchors, annotations, n_classes):
        #TODO tensorize this function for lazyness exploitation
        #TODO rescale bboxes to transformed size
        batch_size = len(annotations)
        regression_targets = np.zeros((batch_size, len(anchors), 4), dtype=np.float32)
        classification_targets = np.zeros((batch_size, len(anchors), n_classes), dtype=np.float32)

        regression_masks = np.zeros((batch_size, len(anchors)), dtype=np.float32)
        classification_masks = np.zeros((batch_size, len(anchors)), dtype=np.float32)
        for i in range(batch_size):
            ann_boxes, ann_labels = annotations[i]['boxes'], annotations[i]['labels']
            assert len(ann_boxes) > 0
            print(ann_boxes)
            ann_boxes = np.array([resize_box_based_on_new_image_size(box, 
                                img_old_size=annotations[i]['image_size'], 
                                img_new_size=self.image_size) for box in ann_boxes])
            fg_bg_mat = foreground_background_matrix(anchors, ann_boxes)
            #fg_bg_mat[anchor_idx, ann_box_idx]
            
            foreground_idxs = np.argwhere(fg_bg_mat==1,)
            background_idxs = np.argwhere(fg_bg_mat==-1,)
            #fg_bg_mat==0 is skipped as paper states
            matched_anchor_idxs = foreground_idxs[:,0]
            matched_ann_box_idxs = foreground_idxs[:,1]
            unmatched_anchor_idxs = background_idxs[:,0]
            #[iou(anch,ann_box)>=0.5 for (anch, ann_box) in zip(anchors[matched_anchor_idxs], ann_boxes[matched_ann_box_idxs, :])].sum()
            regression_targets[i, matched_anchor_idxs, :] = bbox_transform(anchors[matched_anchor_idxs], ann_boxes[matched_ann_box_idxs, :])
            classification_targets[i, matched_anchor_idxs, ann_labels[matched_ann_box_idxs].astype(int)] = 1
            regression_masks[i, matched_anchor_idxs] = 1
            classification_masks[i, np.concatenate((matched_anchor_idxs,unmatched_anchor_idxs))] = 1
                   
        return {"regression_targets": regression_targets, "classification_targets": classification_targets, "regression_masks": regression_masks, "classification_masks": classification_masks}
        
    def train(self):
        NUM = getenv("NUM", 2)
        BS = getenv("BS", 2)
        CNT = getenv("CNT", 10)
        BACKWARD = getenv("BACKWARD", 0)
        TRAINING = getenv("TRAINING", 1)
        ADAM = getenv("ADAM", 0)
        CLCACHE = getenv("CLCACHE", 0)
        backbone = self.model.backbone
        retina = self.model #remember num_classes = 600 for openimages
        #retina.load_from_pretrained()



        anchors_flattened_levels = np.concatenate(retina.anchor_gen(self.image_size))
        params = get_parameters(retina)
        for p in params: p.realize()
        optimizer = optim.SGD(params, lr=0.001)

        Tensor.training = TRAINING
        Tensor.no_grad = not BACKWARD
        for x, annotations in iterate(self.dataset, BS):
            targets = self.get_ground_truths(anchors_flattened_levels, annotations, len(self.dataset.cats.keys()))
            optimizer.zero_grad()
            #self.resize_images_and_bboxes(x, annotations, self.image_size)
            resized = [Image.fromarray(image) for image in x]
            resized = [np.asarray(image.resize(self.image_size)) for image in resized]
            images = Tensor(resized)

            head_outputs = retina(self.input_fixup(images)).numpy()

            loss = self._eltwise_compute_loss(BS, targets, head_outputs) 

    def _eltwise_compute_loss(self, BS, targets, head_outputs):
        total_loss = 0
        for img_idx in range(BS):
            #TODO tensorize, increase mask exploitation, don't use fors
            box_regs = head_outputs[img_idx, :, :4] # anchors??
            cls_preds = head_outputs[img_idx, :, 4:]
            ground_truth_boxes, ground_truth_clss = targets["regression_targets"][img_idx], targets["classification_targets"][img_idx]
            cls_targets_idxs = np.argwhere(targets["classification_masks"][img_idx]==1)
            reg_targets_idxs = np.argwhere(targets["regression_masks"][img_idx]==1)

            box_reg_losses = [smooth_l1_loss(Tensor(box_regs[target_box_idx]), Tensor(ground_truth_boxes[target_box_idx]), beta = 0.11, reduction="sum").numpy() for target_box_idx in reg_targets_idxs]
            focal_losses = [focal_loss(Tensor(cls_preds[target_cls_idx]), Tensor(ground_truth_clss[target_cls_idx])).numpy() for target_cls_idx in cls_targets_idxs]
            #TODO use reductions instead
            total_loss += sum(focal_losses) + sum(box_reg_losses)    
        return total_loss
            
                    

    def resize_images_and_bboxes(self, images, annotations, new_size):
        images = [Image.fromarray(image) for image in images]
        images = [np.asarray(image.resize(new_size)) for image in images]
        for i in range(len(annotations)):
            old_image_size = annotations[i]["image_size"]
            
            x_ratio = old_image_size[0]/new_size[0]
            y_ratio = old_image_size[1]/new_size[1]
            for j in range(len(annotations[i]["boxes"])):
                annotations[i]["boxes"][j][0] /= x_ratio
                annotations[i]["boxes"][j][2] /= x_ratio
                annotations[i]["boxes"][j][1] /= y_ratio
                annotations[i]["boxes"][j][3] /= y_ratio
        
        return images, annotations



    def input_fixup(self,x):
        x = x.permute([0,3,1,2]) / 255.0
        x -= self.input_mean
        x /= self.input_std
        return x



if __name__ == "__main__":
    trainer = RetinaNetTrainer()
    trainer.train()