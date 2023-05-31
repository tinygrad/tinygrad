from extra.utils import get_child
from models.resnet import ResNet
from models.retinanet import ResNetFPN
from torch.hub import load_state_dict_from_url

class MaskRCNN:
  def __init__(self, backbone: ResNet):
    assert isinstance(backbone, ResNet)
    self.backbone = ResNetFPN(backbone)

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    return x

  def load_from_pretrained(self):
    self.url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)