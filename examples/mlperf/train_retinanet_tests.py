from train_retinanet import *
import sys
sys.path.insert(0, r'C:\Users\msoro\Desktop\mlperf\training\single_stage_detector\ssd') # modified for people who don't have 16 CPUs + Nvidia P100 
from model import retinanet as mlp_retinanet


class RetinaNetTrainingInitializer:
    def __init__(self):
        self.model = RetinaNet(ResNeXt50_32X4D(num_classes=None))
        self.reference = mlp_retinanet.retinanet_from_backbone(backbone="resnext50_32x4d",num_classes=self.model.num_classes, image_size = list(IMAGE_SIZES["debug"]), pretrained=False, trainable_backbone_layers=3)
        self.reference.training = True

    def setup(self):
        #self.freeze_spec_backbone_layers()
        self.set_initial_weights(from_mlperf_model=(self.reference is not None))

        Tensor.training = TRAINING
        print("training mode ", Tensor.training)
        Tensor.no_grad = not BACKWARD
        
    
    def set_initial_weights(self, from_mlperf_model=True):
        if from_mlperf_model:
            Warning("Auxiliar weight init")
            sd = get_state_dict(self.model)
            for k,p in dict(self.reference.state_dict()).items():
                assert k in sd.keys()
                sd[k].requires_grad = p.requires_grad

            for k,p in dict(self.reference.named_parameters()).items():
                assert k in sd.keys()
                sd[k].assign(p.clone().detach().numpy())
                sd[k].requires_grad = p.requires_grad
            
            
        else:
            Warning("Strange stuff to be resolved")
            self.set_classification_weights()
            self.set_regression_weights()
            self.set_fpn_weights()


if __name__=="__main__":
    init = RetinaNetTrainingInitializer()
    init.setup()
    trainable = [(k,p) for k,p in get_state_dict(init.model).items() if p.requires_grad]

    breakpoint()