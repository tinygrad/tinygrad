from train_retinanet import *
import sys
sys.path.insert(0, r'C:\Users\msoro\Desktop\mlperf\training\single_stage_detector\ssd') # modified for people who don't have 16 CPUs + Nvidia P100 
from model import retinanet as mlp_retinanet 
from pickle import load as loadp
from model.transform import GeneralizedRCNNTransform as reference_transform

input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
class RetinaNetTrainingInitializer:
    def __init__(self):
        self.model = RetinaNet(ResNeXt50_32X4D(num_classes=None))
        self.reference = mlp_retinanet.retinanet_from_backbone(backbone="resnext50_32x4d",num_classes=self.model.num_classes, image_size = list(IMAGE_SIZES["debug"]), pretrained=False, trainable_backbone_layers=3, data_layout="channels_first")
        self.reference.training = True

    def setup(self):
        #self.freeze_spec_backbone_layers()
        self.set_initial_weights(from_mlperf_model=(self.reference is not None))

        Tensor.training = TRAINING
        print("training mode ", Tensor.training)
        Tensor.no_grad = not BACKWARD
        
    
    def set_initial_weights(self, from_mlperf_model=True):
        Warning("Why are FPN weights assigned but then appear different? Maybe requires_grad")
        if from_mlperf_model:
            Warning("Auxiliar weight init")
            sd = get_state_dict(self.model)
            for k,p in dict(self.reference.state_dict()).items():
                assert k in sd.keys()
                sd[k].requires_grad = p.requires_grad
                sd[k].assign(p.clone().detach().numpy())

            for k,p in dict(self.reference.named_parameters()).items():
                assert k in sd.keys()
                sd[k].assign(p.clone().detach().numpy())
                sd[k].requires_grad = p.requires_grad
            
        else:
            Warning("Strange stuff to be resolved, weights and grads copied directly from MLPerf")
            self.set_classification_weights()
            self.set_regression_weights()
            self.set_fpn_weights()


class RetinaNetWeightsChecker:
    def __init__(self):
        self.init = RetinaNetTrainingInitializer()
        self.reference = self.init.reference
        self.model = self.init.model
        self.init.setup()
        
        
    def check_weight_init(self):
        Warning("50 extra tensors at tg version")
        #FIXME
        model_weights = get_state_dict(self.init.model)
        mlp_weights = self.init.reference.state_dict()
        for item in mlp_weights.keys():
            try:
                assert(torch.allclose(torch_tensor(model_weights[item].numpy()),mlp_weights[item]))
            except AssertionError: breakpoint()
    def check_weight_init_forward(self):
        td, rd = get_state_dict(self.model), self.reference.state_dict()
        assert all(item in td.keys() for item in rd.keys())
        assert all(np.allclose(td[item].numpy(),rd[item].numpy()) for item in rd.keys())

        #with open("random_image.pkl",'rb') as file: sample_image_list = loadp(file)
        sample_image_list = [torch_tensor(np.random.rand(3,200,200)) for _ in range(4)]

        reference_head_outs = self.reference_forward(sample_image_list)
        tg_head_outs = self.model_forward(sample_image_list).numpy()



        Warning("Tinygrad implementation runs sigmoid on cls_logits, mlperf not. Adding sigmoid to mlperf for tests")
        reference_head_cls_logits = torch.sigmoid(reference_head_outs["cls_logits"].detach()).numpy()
        reference_head_bbox_regression = reference_head_outs["bbox_regression"].detach().numpy()
        assert(np.allclose(tg_head_outs[:,:,:4],reference_head_bbox_regression, atol=1e-5))
        assert(np.allclose(tg_head_outs[:,:,4:],reference_head_cls_logits, atol=1e-5))
        print("Equal forward for initial mlperf weights.")

        

    def reference_forward(self, images,):
        #for parallel debugging
        #from model.resnet import ResNet
        reference_sample_image_list, _ = self.reference.transform(images,None)
        reference_input = reference_sample_image_list.tensors.double()
        reference_feature_maps = self.reference.backbone.double()(reference_input) #TODO .double() bothers me. but default usage raises errors ("expected Double instead of Float" bc resnet bias is initialized w/ 32 bits)
        reference_features = list(reference_feature_maps.values())

        self.reference.head.classification_head.__class__.forward = reference_forward_debug_cls
        head_outs = self.reference.head.double().forward(reference_features)
        return head_outs
    
    def model_forward(self, images):
        #from models.resnet import ResNet
        Tensor.training = False
        model_input = self.input_fixup(Tensor(images), normalize=False)
        self.model.head.classification_head.__class__.__call__ = tg_forward_debug_cls
        outs = self.model(model_input)
        return outs

        

    def input_fixup(self,x, normalize = True):
        if normalize: x = x / 255.0
        x -= input_mean
        x /= input_std
        return x


def reference_forward_debug_cls(self,x):
    all_cls_logits = []
    for features in x:
        cls_logits = self.conv(features)
        cls_logits = self.cls_logits(cls_logits)

        # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
        N, _, H, W = cls_logits.shape
        cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
        cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
        cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

        all_cls_logits.append(cls_logits)
    return torch.cat(all_cls_logits, dim=1)

def tg_forward_debug_cls(self, x):
    out = []
    for feat in x:
      
      cls_logits = self.cls_logits(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.num_classes)
      out.append(cls_logits)
      
    return out[0].cat(*out[1:], dim=1).sigmoid()

if __name__=="__main__":
    #init = RetinaNetTrainingInitializer()
    #init.setup()
    #trainable = [(k,p) for k,p in get_state_dict(init.model).items() if p.requires_grad]
    RetinaNetWeightsChecker().check_weight_init()
    RetinaNetWeightsChecker().check_weight_init_forward()
    """rnic = RetinaNetInferenceChecker()
    rf,nf = None,None
    with open("random_image.pkl",'rb') as file: sample_image_list = loadp(file)
    breakpoint()
    sem = Semaphore(1)
    r = rnic.reference_backbone_inference(sample_image_list)
    m = rnic.model_backbone_inference(sample_image_list)
    breakpoint()
    assert(np.allclose(r.numpy()))
    breakpoint()
    i = 0"""
