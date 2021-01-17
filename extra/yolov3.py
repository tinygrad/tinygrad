# https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
# running 

import os
GPU = os.getenv("GPU", None) is not None
import sys
import io
import time
import numpy as np
np.set_printoptions(suppress=True)
from tinygrad.tensor import Tensor
from extra.utils import fetch, get_parameters
from tinygrad.nn import Conv2d, Upsample, EmptyLayer, DetectionLayer, LeakyReLU, BatchNorm2D

from PIL import Image

def imresize(img, w, h):
  return np.array(Image.fromarray(img).resize((w, h)))

def infer(model, img):
  img = np.array(img)
  img = imresize(img, 416, 416)
  img = img[:,:,::-1].transpose((2,0,1))
  img = img[np.newaxis,:,:,:]/255.0
  # Run through model
  prediction = model.forward(Tensor(img))
  return prediction


def parse_cfg(cfg):
  # Return a list of blocks
  #file = open(cfgfile, 'r')
  #lines = file.read().split('\n') # store the lines in a list
  lines = cfg.decode("utf-8").split('\n')
  lines = [x for x in lines if len(x) > 0] # get read of the empty lines 
  lines = [x for x in lines if x[0] != '#'] # get rid of comments
  lines = [x.rstrip().lstrip() for x in lines]

  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[":               # This marks the start of a new block
      if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
        blocks.append(block)     # add it the blocks list
        block = {}               # re-init the block
      block["type"] = line[1:-1].rstrip()     
    else:
      key,value = line.split("=") 
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks

def predict_transform(prediction, inp_dim, anchors, num_classes):
  batch_size = prediction.size(0)
  stride =  inp_dim // prediction.size(2)
  grid_size = inp_dim // stride
  bbox_attrs = 5 + num_classes
  num_anchors = len(anchors)
    
  prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
  prediction = prediction.transpose(1,2).contiguous()
  prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

  anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  #Sigmoid the  centre_X, centre_Y. and object confidencce
  # TODO: Fix this
  prediction[:,:,0] = (prediction[:,:,0].sigmoid())
  prediction[:,:,1] = (prediction[:,:,1].sigmoid())
  prediction[:,:,4] = (prediction[:,:,4].sigmoid())
  """
  prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
  prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
  prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
  """
  #Add the center offsets
  grid = np.arange(grid_size)
  a, b = np.meshgrid(grid, grid)

  x_offset = torch.FloatTensor(a).view(-1,1)
  y_offset = torch.FloatTensor(b).view(-1,1)

  """
  if CUDA:
    x_offset = x_offset.cuda()
    y_offset = y_offset.cuda()
  """

  x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

  prediction[:,:,:2] += x_y_offset

  #log space transform height and the width
  # anchors = torch.FloatTensor(anchors)
  anchors = Tensor(anchors)

  """ TODO: This GPU Stuff
  if CUDA:
    anchors = anchors.cuda()
  """

  anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
  prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

  prediction[:,:,5: 5 + num_classes] = ((prediction[:,:, 5 : 5 + num_classes]).sigmoid())

  prediction[:,:,:4] *= stride

  return prediction


class Darknet:
  def __init__(self, cfg):
    self.blocks = parse_cfg(cfg)
    self.net_info, self.module_list = self.create_modules(self.blocks)
    # Inputs
    print("Modules length:")
    print(len(self.module_list))
    # self.weights = Tensor.uniform(416 * 416)
    # print(self.blocks)

  def create_modules(self, blocks):
    net_info = blocks[0] # Info about model hyperparameters
    prev_filters = 3
    filters = None
    output_filters = []
    module_list = []
    ## module
    for index, x in enumerate(blocks[1:]):
      module_type = x["type"]
      module = []
      if module_type == "convolutional":
        # TODO: BatchNorm2d
        """
        try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        """
        try:
          batch_normalize = int(x["batch_normalize"])
          bias = False
        except:
          batch_normalize = 0
          bias = True

        # layer
        activation = x["activation"]
        filters = int(x["filters"])
        padding = int(x["pad"])
        if padding:
          pad = (int(x["size"]) - 1) // 2
        else:
          pad = 0
        
        conv = Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), pad, bias = True)        
        module.append(conv)

        # BatchNorm2d
        if batch_normalize:
          bn = BatchNorm2D(filters)
          module.append(bn)

        # LeakyReLU activation
        if activation == "leaky":
          module.append(LeakyReLU(0.1))

      elif module_type == "upsample":
        stride = int(x["stride"])
        upsample = Upsample(scale_factor = 2, mode = "bilinear")
        module.append(upsample)
      elif module_type == "route":
        x["layers"] = x["layers"].split(",")
        # Start of route
        start = int(x["layers"][0])
        # End if it exists
        try:
          end = int(x["layers"][0])
        except:
          end = 0
        if start > 0: start = start - index
        if end > 0: end = end - index
        route = EmptyLayer()
        module.append(route)
        if end < 0:
          filters = output_filters[index + start] + output_filters[index + end]
        else:
          filters = output_filters[index + start]
        
      # Shortcut corresponds to skip connection
      elif module_type == "shortcut":
        module.append(EmptyLayer())
      
      elif module_type == "yolo":
        mask = x["mask"].split(",")
        mask = [int(x) for x in mask]

        anchors = x["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
        anchors = [anchors[i] for i in mask]

        detection = DetectionLayer(anchors)
        module.append(detection)
      

      
      # Append to module_list
      module_list.append(module)
      if filters is not None:
        prev_filters = filters
      output_filters.append(filters)
    
    return (net_info, module_list)
  
  def forward(self, x):
    modules = self.blocks[1:]
    outputs = {} # Cached outputs for route layer
    write = 0

    for i, module in enumerate(modules):
      module_type = (module["type"])
      print("Running forward through " + module_type)
      if module_type == "convolutional" or module_type == "upsample":
        for layer in self.module_list[i]:
          x = layer(x)
        # print(self.module_list[i])
        # x = self.module_list[i](x)
      elif module_type == "route":
        layers = module["layers"]
        layers = [int(a) for a in layers]

        if (layers[0]) > 0:
          layers[0] = layers[0] - i
        if len(layers) == 1:
          x = outputs[i + (layers[0])]
        else:
          if (layers[1]) > 0:
            layers[1] = layers[1] - i
          
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]

          x = np.concatenate((map1, map2), 1)
      
      elif module_type == "shortcut":
        from_ = int(module["from"])
        """ Shape mismatch: (1, 64, 410, 410) vs. (1, 64, 412, 412)
        TODO: Implement shortcut
        print("Outputs:")
        print(outputs[i-1].shape)
        print(outputs[i+from_].shape)
        """
        # x = outputs[i - 1] + outputs[i + from_]
      
      elif module_type == "yolo":
        anchors = self.module_list[i][0].anchors
        inp_dim = int(self.net_info["height"])
        num_classes = int(module["classes"])

        # Transform
        x = x.data
        x = predict_transform(x, inp_dim, anchors, num_classes)
        if not write:
          detections = x
          write = 1
        else:
          detections = np.concat((detections, x), 1)
      
      outputs[i] = x
    
    return detections # Return detections


if __name__ == "__main__":
  cfg = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

  # Start model
  model = Darknet(cfg)

  #from PIL import Image
  # url = sys.argv[1]
  url = "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png"
  img = None
  if url.startswith('http'):
    img = Image.open(io.BytesIO(fetch(url)))
  else:
    img = Image.open(url)
  # Predict
  st = time.time()
  print("running inference")
  prediction = infer(model, img)
  print('did inference in %.2f s' % (time.time() - st))
