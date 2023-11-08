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
from extra.training import focal_loss, smooth_l1_loss
from torch import tensor as torch_tensor
import torch
from contextlib import redirect_stdout
from train_retinanet_tests import *
from models.retinanet import decode_bbox
from tinygrad.helpers import dtypes


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

def filter_by_reg_mask(targets):
    res = {}
    allowed_indxs = np.nonzero(targets["regression_masks"])
    res["regression_targets"], res["classification_targets"] = targets["regression_targets"][allowed_indxs], res["classification_targets"][allowed_indxs]
    return res
class RetinaNetTrainer:
    def __init__(self,  model : RetinaNet = RetinaNet(ResNeXt50_32X4D(num_classes=None)), debug = False):
        
        self.model, self.reference = RetinaNetTrainingInitializer().setup(load_debug_weights=True)
        Warning("TODO: at training, ResNet weights should be loaded and most of its layers should be frozen.")
        #self.model.backbone.load_from_pretrained()
        self.input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        self.input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.dataset = COCO(openimages())
        self.coco_eval = COCOeval(self.dataset, iouType="bbox")
        self.debug = debug
        self.image_size = IMAGE_SIZES["debug"] if debug else IMAGE_SIZES["mlperf"]
        self.set_matcher_attributes()
    
    def get_ground_truths(self, anchors, annotations, n_classes):
        #TODO tensorize this function for further lazyness exploitation
        #TODO rescale bboxes to transformed size
        Warning("Check get_ground_truths this is okay, compare ground truth obtainment with MLPERF functions.")
        batch_size = len(annotations)
        regression_targets = np.zeros((batch_size, len(anchors), 4), dtype=np.float32)
        classification_targets = np.zeros((batch_size, len(anchors), n_classes), dtype=np.int32)

        regression_masks = np.zeros((batch_size, len(anchors)), dtype=np.float32)
        classification_masks = np.zeros((batch_size, len(anchors)), dtype=np.float32)
        for i in range(batch_size):
            ann_boxes, ann_labels = annotations[i]['boxes'], annotations[i]['labels']
            assert len(ann_boxes) > 0
            ann_boxes = np.array([resize_box_based_on_new_image_size(box, 
                                img_old_size=annotations[i]['image_size'], 
                                img_new_size=self.image_size) for box in ann_boxes])
            fg_bg_mat = foreground_background_matrix(anchors, ann_boxes)
            #fg_bg_mat[anchor_idx, ann_box_idx]
            foreground_idxs, background_idxs = np.argwhere(fg_bg_mat==1,), np.argwhere(fg_bg_mat==-1,)
            #fg_bg_mat==0 is skipped as paper states
            matched_anchor_idxs, matched_ann_box_idxs  = foreground_idxs[:,0], foreground_idxs[:,1]
            unmatched_anchor_idxs = background_idxs[:,0]
            
            regression_targets[i, matched_anchor_idxs, :] = bbox_transform(anchors[matched_anchor_idxs], ann_boxes[matched_ann_box_idxs, :])
            classification_targets[i, matched_anchor_idxs, ann_labels[matched_ann_box_idxs].astype(int)] = 1
            regression_masks[i, matched_anchor_idxs] = 1
            classification_masks[i, np.concatenate((matched_anchor_idxs,unmatched_anchor_idxs))] = 1
                   
        #TODO decode_bbox(regression_targets[0],anchors) runs

        return {
            
            #FIXME you can directly pass anchors, just wanted to check decode function
            "decoded_regression_targets": np.array([decode_bbox(img_reg_target, anchors) for img_reg_target in regression_targets])
                ,"regression_targets": regression_targets, "classification_targets": classification_targets,
                "regression_masks": regression_masks, "classification_masks": classification_masks}
    
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
    
    def train(self):
        #self.check_weight_init_forward() FIXME this works but may modify gradients

        retina, anchors_orig, anchors_flattened_levels, optimizer = self.tg_init_setup()
        
        self.reference.train()
        
        checker = None

        for x, annotations in iterate(self.dataset, BS):
            Warning("Annotations not being resized for MLPERF reference model")
            if len(sys.argv)>2 and sys.argv[2]=="old_loss": precomp_tg_targets = self.get_ground_truths(anchors_flattened_levels, annotations, len(self.dataset.cats.keys()))
            resized = [Image.fromarray(image) for image in x]
            resized = [np.asarray(image.resize(self.image_size)) for image in resized]

            images = self.input_fixup(Tensor(resized, requires_grad=False))
            reference_images = torch.permute(torch.from_numpy(np.array(resized)),(0,3,1,2))/255 


            if sys.argv[1]=='m':
                model_head_outputs = retina(images)

            if sys.argv[1]=='r':
                reference_loss = self.reference_forward(reference_images.double(),annotations=annotations)
                reference_head_outputs = self.reference.last_head_outputs
            Warning("reference outputs may be losses instead of head outputs if model is being trained")

            
            if len(sys.argv)>2 and sys.argv[2]=="old_loss": imgwise_loss = self._imgwise_compute_loss(BS, precomp_tg_targets, model_head_outputs)
            else:
                self.dataset_annotations_to_tg(annotations)
                tg_anchors_flattened_levels = [Tensor(anchors_flattened_levels.astype("float32")) for _ in range(BS)]
                self.compute_loss_tg(annotations, model_head_outputs, tg_anchors_flattened_levels)
            #if self.debug: checker.check_losses(imgwise_loss)
            optimizer.zero_grad()
            imgwise_loss.backward()
            optimizer.step()
            
            #TODO input_size=self.image_size? WATCH OUT PARAMETERS AND USAGE
            if self.debug:
                predictions = retina.postprocess_detections(model_head_outputs.numpy(),input_size=self.image_size,anchors=anchors_orig,orig_image_sizes=[t["image_size"] for t in annotations])

                for image, image_preds in zip(x, predictions):
                    img = Image.fromarray(image)
                    draw = ImageDraw.Draw(img)
                    for box in image_preds["boxes"]:
                        draw.rectangle([box[0], box[1],box[0]+box[2], box[1]+box[3]], outline="red")
                    img.show()

                #TODO this should be done on validation split predictions (no grad)
                self.mAP_compute(annotations, predictions)

    def tg_init_setup(self, mlperf_model = None):
        retina = self.model
        #self.freeze_spec_backbone_layers()
        #self.set_initial_weights(from_mlperf_model=(self.reference is not None))
        
        anchors_orig = retina.anchor_gen(self.image_size) #TODO: get them just with reshape of flattened?
        anchors_flattened_levels = np.concatenate(anchors_orig)
        optimizer = optim.SGD(get_parameters(retina), lr=0.001)

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

    def _batchwise_compute_loss(self, BS, targets, head_outputs):
        #TODO is this possible with tensor mul only?
        total_loss = Tensor(0)
        box_regs = head_outputs[:, :, :4]
        cls_preds = head_outputs[:, :, 4:]
        ground_truth_boxes, ground_truth_clss = targets["regression_targets"], targets["classification_targets"]
        cls_targets_idxs = np.argwhere(targets["classification_masks"]==1)
        reg_targets_idxs = np.argwhere(targets["regression_masks"]==1)

        box_reg_losses = smooth_l1_loss(Tensor(box_regs[reg_targets_idxs], requires_grad=True), Tensor(ground_truth_boxes[reg_targets_idxs]), beta = 0.11, reduction="sum")
        #TODO
        Warning("You are normalizing over the BATCH number of anchors, not each image loss by its number of anchors...")
        focal_losses = focal_loss(Tensor(cls_preds[cls_targets_idxs], requires_grad=True).sigmoid(), Tensor(ground_truth_clss[cls_targets_idxs]), reduction="sum").div(len(cls_preds))
        total_loss += focal_losses + box_reg_losses
        print(total_loss.numpy() , " batch loss")
        return total_loss
    
    def _imgwise_compute_loss(self, BS, targets, head_outputs):
        from torchvision.ops.focal_loss import sigmoid_focal_loss
        from torch.nn import SmoothL1Loss
        from torch import tensor
        regression_losses = []
        classification_losses = []
        for img_idx in range(BS):
            #TODO tensorize even more for BS level, increase mask exploitation, don't use fors
            box_regs = head_outputs[img_idx, :, :4]
            cls_preds = head_outputs[img_idx, :, 4:]
            ground_truth_boxes, ground_truth_clss = targets["regression_targets"][img_idx], targets["classification_targets"][img_idx]

            def generate_mtx_mask(targs_mask, preds):
                #DEBUG and replace below
                return Tensor(targs_mask).reshape(targs_mask.shape[0],1).expand(targs_mask.shape[0],targs_mask.shape[1])
            reg_mask_mtx = Tensor(targets["regression_masks"][img_idx], requires_grad=False).reshape(box_regs.shape[0],1).expand(box_regs.shape[0],4)
            cls_mask_mtx = Tensor(targets["classification_masks"][img_idx], requires_grad=False).reshape(cls_preds.shape[0],1).expand(cls_preds.shape[0],self.model.num_classes)

            focal_losses = focal_loss((cls_preds * cls_mask_mtx).sigmoid(), Tensor(ground_truth_clss, requires_grad=False) * cls_mask_mtx, reduction="sum").div(cls_preds.shape[0])
            #TODO box regs are very sparse, maybe matmul makes training slower?
            box_reg_losses = smooth_l1_loss(box_regs * reg_mask_mtx, Tensor(ground_truth_boxes, requires_grad=False) * reg_mask_mtx, beta = 0.11, reduction="sum")

            regression_losses.append(box_reg_losses)
            classification_losses.append(focal_losses)

        classification_losses = Tensor.stack(classification_losses).sum()
        regression_losses = Tensor.stack(regression_losses).sum()
        print((classification_losses + regression_losses).numpy() , " batch loss")

        return classification_losses + regression_losses
    
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
            breakpoint()
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def _imgwise_compute_loss_np(self, BS, targets, head_outputs):
        from torchvision.ops.focal_loss import sigmoid_focal_loss
        from torch.nn import SmoothL1Loss
        from torch import tensor
        total_loss = Tensor(0)
        for img_idx in range(BS):
            #TODO tensorize, increase mask exploitation, don't use fors
            box_regs = head_outputs[img_idx, :, :4].numpy()
            cls_preds = head_outputs[img_idx, :, 4:].numpy()
            ground_truth_boxes, ground_truth_clss = targets["regression_targets"][img_idx], targets["classification_targets"][img_idx]
            cls_targets_idxs = np.argwhere(targets["classification_masks"][img_idx]==1).flatten()
            reg_targets_idxs = np.argwhere(targets["regression_masks"][img_idx]==1).flatten()

            #TODO this fails when there are 0 regression targets
            #TODO is zero regression targets possible?
            box_reg_losses = smooth_l1_loss(Tensor(box_regs[reg_targets_idxs]), Tensor(ground_truth_boxes[reg_targets_idxs]), beta = 0.11, reduction="sum")
            focal_losses = focal_loss(Tensor(cls_preds[cls_targets_idxs]).sigmoid(), Tensor(ground_truth_clss[cls_targets_idxs]), reduction="sum").div(len(cls_preds))
            #TODO use reductions instead
            total_loss += focal_losses + box_reg_losses
        print(total_loss.numpy() , " batch loss")
        return total_loss
    
    def _eltwise_compute_loss(self, BS, targets, head_outputs):
        total_loss = 0
        for img_idx in range(BS):
            #TODO tensorize, increase mask exploitation, don't use fors
            box_regs = head_outputs[img_idx, :, :4]
            cls_preds = head_outputs[img_idx, :, 4:]
            ground_truth_boxes, ground_truth_clss = targets["regression_targets"][img_idx], targets["classification_targets"][img_idx]
            cls_targets_idxs = np.argwhere(targets["classification_masks"][img_idx]==1)
            reg_targets_idxs = np.argwhere(targets["regression_masks"][img_idx]==1)

            box_reg_losses = [smooth_l1_loss(Tensor(box_regs[target_box_idx]), Tensor(ground_truth_boxes[target_box_idx]), beta = 0.11, reduction="sum").numpy() for target_box_idx in reg_targets_idxs]
            focal_losses = [focal_loss(Tensor(cls_preds[target_cls_idx]).sigmoid(), Tensor(ground_truth_clss[target_cls_idx])).numpy() for target_cls_idx in cls_targets_idxs]
            #TODO use reductions instead
            total_loss += sum(focal_losses) + sum(box_reg_losses)    
        return total_loss
            
    def _dummy_test_sigmoid_focal_loss(self):
        from torchvision.ops.focal_loss import sigmoid_focal_loss
        from torch import tensor
        from tinygrad.tensor import sigmoid
        from torch.nn import SmoothL1Loss

        a_, b_ = Tensor([[0.7, 0.2,0.1],[0.1,0.5,0.4],[0.3,0.1,0.6]]), Tensor([[1,0,0],[0,0,1],[0,0,1]])
        pt_loss = sigmoid_focal_loss(torch_tensor(a_.numpy()),torch_tensor(b_.numpy()), reduction="sum")
        tg_loss = focal_loss(a_,b_, reduction="none").numpy().sum()


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


    def box_iou(self,boxes1 : Tensor,boxes2:Tensor) -> Tuple[Tensor,Tensor]:
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

    def proposal_matcher(self, match_quality_matrix : Tensor):
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
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches
    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
        Warning("Find an efficient way to make argwhere or similar without numpy conversion")
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
        
    def box_area(self, boxes : Tensor) -> Tensor:
        boxes = self._upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def _upcast(self,t:Tensor) -> Tensor:
        if t.is_floating_point():
            return t if t.dtype in (dtypes.float32, ) else t.float()
        else:
            Warning(".int not implemented, maybe not that useful but this is used in reference")
            return t #if t.dtype in (dtypes.int32,) else t.int()
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
    trainer = RetinaNetTrainer(debug=True)
    trainer.train()