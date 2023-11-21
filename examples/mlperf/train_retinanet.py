from typing import Any
from extra.datasets import openimages
from PIL import Image, ImageDraw
from extra.datasets.openimages import openimages, iterate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tinygrad.nn import optim
from tinygrad.state import get_parameters
from tqdm import trange
from typing import List, Tuple
from extra.training import l1_loss,focal_loss
from torch import tensor as torch_tensor
import torch
from contextlib import redirect_stdout
from train_retinanet_tests import *
from models.retinanet import decode_bbox
from tinygrad.helpers import dtypes, Context
import pickle as pkl
from numba import njit





def focal_loss_np(p,targets,alpha: float = 0.25,gamma: float = 2,reduction: str = "none",):

    p , targets = p.astype(np.float32) , targets.astype(np.float32)
    p_t = p * targets + (1 - p) * (1 - targets)
    ce_loss = -np.log(p_t + 1e-10)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def l1_loss_np(input, target, reduction: str = "none"):
    
    
    loss = np.abs(input - target)
    


    if reduction == "mean":
        return loss.mean() if loss.size > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        return loss.sum()
    return loss

def encode_single(reference_boxes, proposals,weights=(1.0, 1.0, 1.0, 1.0)):
        
        dtype = reference_boxes.dtype
        weights = np.array(weights, dtype=dtype)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

@njit
def encode_boxes(reference_boxes, proposals, weights):
    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]


    proposals_x1 = np.expand_dims(proposals[:, 0],axis=1)
    proposals_y1 = np.expand_dims(proposals[:, 1],axis=1)
    proposals_x2 = np.expand_dims(proposals[:, 2],axis=1)
    proposals_y2 = np.expand_dims(proposals[:, 3],axis=1)

    reference_boxes_x1 = np.expand_dims(reference_boxes[:, 0],axis=1)
    reference_boxes_y1 = np.expand_dims(reference_boxes[:, 1],axis=1)
    reference_boxes_x2 = np.expand_dims(reference_boxes[:, 2],axis=1)
    reference_boxes_y2 = np.expand_dims(reference_boxes[:, 3],axis=1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.concatenate((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets

def _sum(x: List[np.ndarray]) -> np.ndarray:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

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


def annotations_to_mlperf_targets(annotations : dict[str,Tensor]) -> List[dict[str,torch.tensor]]:
    for img_annotations in annotations:
        img_annotations["boxes"] = torch_tensor(img_annotations["boxes"])
        img_annotations["labels"] = torch_tensor(img_annotations["labels"])
    """assert(set(['regression_targets', 'classification_targets']).issubset(set(tg_targets.keys())))
    assert tg_targets['regression_targets'].shape[:2]==tg_targets['classification_targets'].shape[:2]
    allowed_indxs = np.nonzero(tg_targets["regression_masks"])
    targets = []
    for img_regs,img_cls_mask in zip(tg_targets['decoded_regression_targets'][allowed_indxs],tg_targets['classification_targets'][allowed_indxs]):
        img_cls_idxs = np.where(img_cls_mask)
        d_img = {'boxes':torch_tensor(np.array([img_regs])) if len(img_regs.shape)<=1 else torch_tensor(img_regs), 
                 'labels':torch_tensor(np.array([img_cls_idxs])) if len(img_cls_mask.shape)<=1 else torch_tensor(img_cls_idxs)}
        targets.append(d_img)"""
    #return targets

def _save_object(obj, filename):
    #For parallel read-write debugging with reference training
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)

# Function to load an object from a file using pickle
def _load_object(filename):
    #For parallel read-write debugging with reference training
    with open(filename, 'rb') as f:
        return pkl.load(f)

class RetinaNetTrainer:
    def __init__(self,  model : RetinaNet = RetinaNet(ResNeXt50_32X4D(num_classes=None)), debug = False):
        
        self.model, self.reference = RetinaNetTrainingInitializer().setup(load_debug_weights=False)
        self.model.head.compute_loss = self.head_loss_fn
        self.model.head.classification_head.compute_loss = self.cls_loss_fn
        self.model.head.regression_head.compute_loss = self.reg_loss_fn
        
        #self.model.backbone.load_from_pretrained()
        self.input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.dataset = COCO(openimages())
        self.coco_eval = COCOeval(self.dataset, iouType="bbox")
        self.debug = debug
        
        self.image_size = IMAGE_SIZES["debug"] if debug else IMAGE_SIZES["mlperf"]
        Warning("setting sizes to tiny")
        self.image_size = IMAGE_SIZES["debug"]
        self.set_matcher_attributes()
        
    def freeze_spec_backbone_layers(self, layers_to_train = ["layer2", "layer3", "layer4"]):
        """(MLPerf) The weights of the first two stages are frozen (code). 
        In addition, all batch norm layers in the backbone are frozen (code)."""
        for (key,val) in get_state_dict(self.model.backbone).items():
            if any([layer in key for layer in layers_to_train]):
                val.requires_grad = True
            else:
                val.requires_grad = False
            if("bn" in key):
                val.requires_grad = False
            
    def reference_forward(self, images,annotations=None):
        annotations_to_mlperf_targets(annotations)
        Warning("Still adapting from train_retinanet_tests-.py")    
        training_forward_outs = self.reference.double()(images, annotations)
        return training_forward_outs
    
    def _reference_forward_raw(self, images,):
        #for parallel debugging
        #from model.resnet import ResNet
        reference_sample_image_list, _ = self.reference.transform(images,None)
        reference_input = reference_sample_image_list.tensors.double()
        reference_feature_maps = self.reference.backbone.double()(reference_input) #TODO .double() bothers me. but default usage raises errors ("expected Double instead of Float" bc resnet bias is initialized w/ 32 bits)
        reference_features = list(reference_feature_maps.values())

        head_outs = self.reference.head.double().forward(reference_features)
        return head_outs
    def _model_forward_raw(self, images):
        def _input_fixup(x, normalize = True):
            if normalize: x = x / 255.0
            x -= input_mean
            x /= input_std
            return x
        #from models.resnet import ResNet
        Tensor.training = False
        model_input = _input_fixup(Tensor(images), normalize=False)
        self.model.head.classification_head.__class__.__call__ = tg_forward_debug_cls
        outs = self.model(model_input)
        return outs

    def check_weight_init_forward(self):
        td, rd = get_state_dict(self.model), self.reference.state_dict()
        assert all(item in td.keys() for item in rd.keys())
        assert all(np.allclose(td[item].numpy(),rd[item].numpy()) for item in rd.keys())

        #with open("random_image.pkl",'rb') as file: sample_image_list = loadp(file)
        sample_image_list = [torch_tensor(np.random.rand(3,200,200)) for _ in range(4)]

        reference_head_outs = self._reference_forward_raw(sample_image_list)
        tg_head_outs = self._model_forward_raw(sample_image_list).numpy()



        Warning("Tinygrad implementation runs sigmoid on cls_logits, mlperf not. Adding sigmoid to mlperf for tests")
        reference_head_cls_logits = torch.sigmoid(reference_head_outs["cls_logits"].detach()).numpy()
        reference_head_bbox_regression = reference_head_outs["bbox_regression"].detach().numpy()
        assert(np.allclose(tg_head_outs[:,:,:4],reference_head_bbox_regression, atol=1e-5))
        assert(np.allclose(tg_head_outs[:,:,4:],reference_head_cls_logits, atol=1e-5))
        print("Equal forward for initial mlperf weights.")

    def model_forward(self, images):
        def input_fixup(x, normalize = True):
            #FIXME dynamic function definition may slow things up
            if normalize: x = x / 255.0
            x -= input_mean
            x /= input_std
            return x
        #from models.resnet import ResNet
        Tensor.training = False
        model_input = input_fixup(Tensor(images), normalize=False)
        outs = self.model(model_input)
        Tensor.training = TRAINING
        return outs
    
    def resize_bbox(self,annotations):
        for annotations_by_image in annotations:
            orig_h, orig_w = annotations_by_image["image_size"]
            h,w = self.image_size
            h_ratio, w_ratio = h/orig_h, w/orig_w
            annotations_by_image["boxes"][:,0] *= w_ratio
            annotations_by_image["boxes"][:,1] *= h_ratio
            annotations_by_image["boxes"][:,2] *= w_ratio
            annotations_by_image["boxes"][:,3] *= h_ratio
            annotations_by_image["image_size"] = self.image_size
        return annotations
                
    def train_one_epoch(self):
        import time
        from copy import deepcopy
        #self.check_weight_init_forward() FIXME this works but may modify gradients
        retina, anchors_orig, anchors_flattened_levels, optimizer = self.tg_init_setup()
        anchors_flattened_levels = [anchors_flattened_levels.astype("float32") for _ in range(BS)]
        
        self.reference.train()
        ref_optimizer = torch.optim.Adam([p for p in self.reference.parameters() if p.requires_grad], lr=0.0001)


        for x, annotations in iterate(self.dataset, BS):
            Warning("Annotations not being resized for MLPERF reference model")
            breakpoint()
            resized = [Image.fromarray(image) for image in x]
            resized = [np.asarray(image.resize(self.image_size)) for image in resized]

            orig_annotations = deepcopy(annotations)
            annotations = self.resize_bbox(annotations)

            images = self.input_fixup(Tensor(resized, requires_grad=False))
            reference_images = torch.permute(torch.from_numpy(np.array(resized)),(0,3,1,2))/255 


            if sys.argv[1]=='m':
                model_head_outputs = retina(images)
                model_head_outputs = {'cls_logits' : model_head_outputs[:,:,4:], 
                                    'bbox_regression' : model_head_outputs[:,:,:4]}
            reference_loss, model_loss = None, None
            if sys.argv[1]=='r':
                reference_loss = self.reference_forward(reference_images.double(),annotations=annotations)
                reference_head_outputs = self.reference.last_head_outputs
            Warning("reference outputs may be losses instead of head outputs if model is being trained")

            
            if sys.argv[1]=='m':
                #self.dataset_annotations_to_tg(annotations)
                model_loss = self.compute_loss(annotations, model_head_outputs, anchors_flattened_levels)
            
            

            optimizer.zero_grad()
            if sys.argv[1]=='m':
                losses = model_loss["classification"]+model_loss["bbox_regression"]
                losses.backward()
                print(losses.numpy())
                optimizer.step()
            elif sys.argv[1]=='r':
                losses = reference_loss["classification"]+reference_loss["bbox_regression"]
                losses.backward()
                print(losses)
                ref_optimizer.step()
            
            time.sleep(0.2)
            
            
            #TODO input_size=self.image_size? WATCH OUT PARAMETERS AND USAGE
            if self.debug:
                predictions = retina.postprocess_detections(model_head_outputs.numpy(),input_size=self.image_size,anchors=anchors_orig,orig_image_sizes=[t["image_size"] for t in annotations])
                #TODO this should be done on validation split predictions (no grad)
                self.mAP_compute(annotations, predictions)

    def tg_init_setup(self, mlperf_model = None):
        retina = self.model
        
        anchors_orig = retina.anchor_gen(self.image_size) #TODO: get them just with reshape of flattened?
        anchors_flattened_levels = np.concatenate(anchors_orig)
        optimizer = optim.SGD(get_parameters(retina), lr=0.0001)

        Tensor.training = TRAINING
        print("training mode ", Tensor.training)
        Tensor.no_grad = not BACKWARD
        return retina,anchors_orig,anchors_flattened_levels,optimizer
            
    def set_initial_weights(self, from_mlperf_model = False):
        if from_mlperf_model:
            sd = get_state_dict(self.model)
            for k,p in self.reference.named_parameters().items():
                assert k in sd.keys()
                sd[k] = Tensor(p.clone().numpy())
                sd[k].requires_grad = p.requires_grad
        else:
            Warning("Strange stuff to be resolved")
            self.set_classification_weights()
            self.set_regression_weights()
            self.set_fpn_weights()

    def set_classification_weights(self,mean : float = 0, std : float = 0.01, conv_bias : float = -4.59511985013459, bias : float = 0):
        for (key,val) in get_state_dict(self.model.head.classification_head).items():
            if "weight" in key and val.requires_grad: 
                val = Tensor.normal(*(val.shape), mean=mean, std=std)
            elif "bias" in key and val.requires_grad: 
                val = Tensor.full(*(val.shape), conv_bias)
        

    def set_regression_weights(self, mean : float =0, std : float =0.01, conv_bias : float = 0):
        for (key,val) in get_state_dict(self.model.head.regression_head).items():
            if "weight" in key and val.requires_grad: 
                #print(key)
                val = Tensor.normal(*(val.shape), mean=mean, std=std)
            elif "bias" in key and val.requires_grad:
                #print(key)
                val = Tensor.full(*(val.shape), conv_bias)
    
    def set_fpn_weights(self, bias : float =0):
        """ 
        The FPN network weights are initialized with uniform Kaiming (also known as He initialization) 
        using negative slope=1. 
        The biases are initialized with zeros (code).
        """
        for (key,val) in get_state_dict(self.model.backbone).items():
            if "conv" in key and val.requires_grad:
                #print(key) 
                val = Tensor.kaiming_uniform(*(val.shape),a=1)
            elif "bias" in key and val.requires_grad:
                #print(key)
                if val.__hash__ == (self.model.backbone.body.bn1.bias).__hash__: 
                    print("This should not happen")
                val = Tensor.full(*(val.shape), bias)
        #mlp_fpn = [(name,param) for name,param in mlperf_model.backbone.fpn.named_parameters()]
        #tg_w = [(name,param.numpy()) for name,param in get_state_dict(self.model.backbone.fpn).items()]
        
    def dataset_annotations_to_tg(self, np_annotations):
        for img_annotations in np_annotations:
            img_annotations["boxes"] = Tensor(img_annotations["boxes"])
            img_annotations["labels"] = Tensor(img_annotations["labels"])
        return
    def mAP_compute(self, annotations, predictions):
        coco_results  = [{"image_id": annotations[i]["image_id"], "category_id": label, "bbox": box.tolist(), "score": score}
      for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())]
        coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(self.coco_eval.params.catIds), len(self.coco_eval.params.areaRng)
        img_ids = [t["image_id"] for t in annotations]
        with redirect_stdout(None):
            self.coco_eval.cocoDt = self.dataset.loadRes(coco_results)
            self.coco_eval.params.imgIds = img_ids
            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
        evaluated_imgs.extend(img_ids)
        coco_evalimgs.append(np.array(self.coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
    
    
    
    def head_loss_fn(self, targets, head_outputs, anchors, matched_idxs):
        return {
            'classification': self.model.head.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.model.head.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }
    def cls_loss_fn(self, targets, head_outputs, matched_idxs):
        losses = []

        cls_logits = head_outputs['cls_logits']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = np.zeros(cls_logits_per_image.shape, dtype=np.float32)

            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            # compute the classification loss

            #TODO maybe using mul by mask may be more efficient than gather, depends on sparsity
            #FIXME ^
            losses.append(focal_loss(
                cls_logits_per_image.gather(Tensor(np.argwhere(valid_idxs_per_image).flatten()),0),
                Tensor(gt_classes_target[valid_idxs_per_image],device=cls_logits_per_image.device,requires_grad=False),
                reduction='sum',
            ) / max(1, num_foreground))

        return _sum(losses) / len(targets)
    def reg_loss_fn(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = np.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.size

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]

            bbox_regression_per_image = bbox_regression_per_image.gather(Tensor(foreground_idxs_per_image.flatten()),0)
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            # compute the regression targets
            target_regression = encode_single(matched_gt_boxes_per_image, anchors_per_image).astype(np.float32)
            # compute the loss
            losses.append(l1_loss(
                bbox_regression_per_image,
                Tensor(target_regression,requires_grad=False,device=bbox_regression_per_image.device),
                reduction='sum'
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))
    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].size == 0:
                matched_idxs.append(np.full((anchors_per_image.size(0),), -1, dtype=np.int64))
                continue

            match_quality_matrix = self.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        return self.model.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def box_iou(self,boxes1, boxes2):
        inter, union = self._box_inter_union(boxes1, boxes2)
        iou = inter / union
        return iou

    def _box_inter_union(self,boxes1, boxes2):
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = self._upcast(rb - lt)
        wh = np.clip(wh, 0, None)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        return inter, union
    def box_area(self,boxes):
        boxes = self._upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    def _upcast(self,t):
        # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
        if np.issubdtype(t.dtype, np.floating):
            return t if t.dtype in (np.float32, np.float64) else t.astype(np.float32)
        else:
            return t if t.dtype in (np.int32, np.int64) else t.astype(np.int32)
    
    def compute_loss_tg(self, targets, head_outputs, anchors):
        #FIXME making tinygrad version
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        Warning("Assuming fixed anchors (all images' channels are resized to 800x800 in the benchmark). No anchor redundant compute.")
        Warning("You might want to implement this in numpy before the return compute_loss and then compare performances")

        matched_idxs = []
        
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(Tensor.full((anchors_per_image.size(0),), -1, dtype=dtypes.int64,
                                               device=anchors_per_image.device))
                continue

            match_quality_matrix = self.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)
    
    
    def _dummy_test_sigmoid_focal_loss(self):
        from torchvision.ops.focal_loss import sigmoid_focal_loss
        from torch import tensor
        from tinygrad.tensor import sigmoid
        from torch.nn import SmoothL1Loss

        a_, b_ = Tensor([[0.7, 0.2,0.1],[0.1,0.5,0.4],[0.3,0.1,0.6]]), Tensor([[1,0,0],[0,0,1],[0,0,1]])
        pt_loss = sigmoid_focal_loss(torch_tensor(a_.numpy()),torch_tensor(b_.numpy()), reduction="sum")
        tg_loss = focal_loss(a_,b_, reduction="none").numpy().sum()


    def box_iou_tg(self,boxes1 : Tensor,boxes2:Tensor) -> Tuple[Tensor,Tensor]:
            area1 = self.box_area(boxes1)
            area2 = self.box_area(boxes2)


            lt = boxes1[:, None, :2].maximum(boxes2[:, :2])  # [N,M,2]
            rb = boxes1[:, None, 2:].minimum(boxes2[:, 2:])  # [N,M,2]
            wh = self._upcast(rb - lt).clip(min_=0,max_=float('inf') )  # [N,M,2]
            inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
            union = area1[:, None] + area2 - inter
            iou = inter / union
            return iou

    def set_matcher_attributes(self):
        self.allow_low_quality_matches = True
        self.low_threshold = 0.4
        self.high_threshold = 0.5
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
    def proposal_matcher(self,match_quality_matrix):
        if match_quality_matrix.size == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        matched_vals = np.amax(match_quality_matrix, axis=0)
        matches = np.argmax(match_quality_matrix, axis=0)

        if self.allow_low_quality_matches:
            all_matches = np.copy(matches)
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold

        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )

        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD


        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):

        highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
        gt_pred_pairs_of_highest_quality = np.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]

        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

    def proposal_matcher_tg(self, match_quality_matrix : Tensor):
        sink_dir = "./examples/mlperf/reference_objects/matched_vals/"
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        
        #FIXME this works different from reference

        Warning("Find an efficient way to find match_quality_matrix max,idxs without numpy conversion")
        #FIXME
        
        matched_vals, matches = Tensor(match_quality_matrix.numpy().max(axis=0)),Tensor(match_quality_matrix.numpy().argmax(axis=0))
        #maybe  match_quality_matrix.max()

        if self.allow_low_quality_matches:
            all_matches = Tensor.zeros_like(matches).assign(matches)
        else:
            all_matches = None
        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) * (
            matched_vals < self.high_threshold
        )
        
        
        matches = Tensor.full_like(matches,self.BELOW_LOW_THRESHOLD) * (below_low_threshold) + matches * (1-below_low_threshold)

        matches = Tensor.full_like(matches,self.BETWEEN_THRESHOLDS) * (between_thresholds) + matches * (1-between_thresholds)
        
        
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_tg(matches, all_matches, match_quality_matrix)

        return matches
    def set_low_quality_matches_tg(self, matches, all_matches, match_quality_matrix):
        sink_dir = "./examples/mlperf/reference_objects/matched_vals/"

        
        highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
        Warning("To be tested and optimized. Find an efficient way to make argwhere or similar without numpy conversion.")
        #FIXME

        gt_pred_pairs_of_highest_quality = np.nonzero(
            match_quality_matrix.numpy() == highest_quality_foreach_gt[:, None].numpy()
        )
        gt_pred_pairs_of_highest_quality = list(gt_pred_pairs_of_highest_quality)
        for i in range(len(gt_pred_pairs_of_highest_quality)):gt_pred_pairs_of_highest_quality[i] = Tensor(gt_pred_pairs_of_highest_quality[i])
        gt_pred_pairs_of_highest_quality = tuple(gt_pred_pairs_of_highest_quality)


        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]


        Warning("Find an efficient way to make argwhere or similar without numpy conversion")
        #FIXME
        matches = matches.numpy()
        all_matches = all_matches.numpy()
        pred_inds_to_update = pred_inds_to_update.numpy()

        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

        matches = Tensor(matches)
        
        all_matches = Tensor(all_matches)
        
    def box_area_tg(self, boxes : Tensor) -> Tensor:
        boxes = self._upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    
    def input_fixup(self,x):
        x = x.permute([0,3,1,2]) / 255.0
        x -= self.input_mean
        x /= self.input_std
        return x

class RetinaNetMlPerfTrainingChecker:
    def __init__(self, tg_model : RetinaNet) -> None:
        import sys
        sys.path.insert(0, r'C:\Users\msoro\Desktop\mlperf\training\single_stage_detector\ssd') # modified for people who don't have 16 CPUs + Nvidia P100 
        from model import retinanet as mlp_retinanet

        self.trainer = RetinaNetTrainer(model=tg_model,debug=True)
        self.model = tg_model
        self.mlperf_model = mlp_retinanet.retinanet_from_backbone(backbone="resnext50_32x4d",num_classes=tg_model.num_classes, image_size = list(IMAGE_SIZES["debug"]), pretrained=False, trainable_backbone_layers=3)

    def check_anchorgen(self):
        #TODO refactor. Make more robust (can it?)
        model, anchors_orig, anchors_flattened_levels, optimizer = self.trainer.tg_init_setup()
        cell_anchors = self.mlperf_model.anchor_generator.cell_anchors
        self.mlperf_model.training = False
        sample_image_list = [torch_tensor(np.random.rand(3,200,200))]
        sample_image_list, _ = self.mlperf_model.transform(sample_image_list,None)

        feature_maps = self.mlperf_model.backbone.double()(sample_image_list.tensors.double()) #TODO .double() bothers me. but default usage raises errors ("expected Double instead of Float" bc resnet bias is initialized w/ 32 bits)
        
        features = list(feature_maps.values())
        anchors_one_image = self.mlperf_model.anchor_generator(sample_image_list, features)
        self.mlperf_model.training = True
        assert torch.equal(torch_tensor(anchors_flattened_levels),anchors_one_image[0])

    



if __name__ == "__main__":
    trainer = RetinaNetTrainer(debug=False)
    trainer.train_one_epoch()