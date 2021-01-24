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

def temp_process_results(prediction, confidence = 0.9, num_classes = 80):
  coco_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
  coco_labels = coco_labels.decode('utf-8').split('\n')

  prediction = prediction.detach().cpu().data

  conf_mask = (prediction[:,:,4] > confidence)
  conf_mask = np.expand_dims(conf_mask, 2)
  prediction = prediction * conf_mask

  img = prediction[0]

  # Loop through the probabilities for all bounding boxes, smh.
  # img => (22743, 85)
  # 22743 bounding boxes, with probabilities for 80 class labels, and 5 bbox attributes
  # Ignore bbox attributes, and just show highest probabilities
  def max_argmax(input, dim):
    return np.amax(input, axis=dim), np.argmax(input, axis=dim)
  
  detections = {}

  for i in range(img.shape[0]):
    labels = img[i][5:num_classes + 5]
    index = np.argmax(labels)
    probability = img[i][index]
    if probability > confidence:
      if coco_labels[index] in detections.keys():
        detections[coco_labels[index]] += 1
      else:
        detections[coco_labels[index]] = 1
  
  print("Detections, the more the bigger probability")
  detections = dict(sorted(detections.items(), key=lambda item: item[1]))
  print(detections)

def bbox_iou(box1, box2):
  """
  Returns the IoU of two bounding boxes
  IoU: IoU = Area Of Overlap / Area of Union -> How close the predicted bounding box is
  to the ground truth bounding box. Higher IoU = Better accuracy

  In training, used to track accuracy. with inference, using to remove duplicate bounding boxes
  """
  # Get the coordinates of bounding boxes
  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

  # get the corrdinates of the intersection rectangle
  inter_rect_x1 = torch.max(b1_x1, b2_x1)
  inter_rect_y1 = torch.max(b1_y1, b2_y1)
  inter_rect_x2 = torch.min(b1_x2, b2_x2)
  inter_rect_y2 = torch.min(b1_y2, b2_y2)

  #Intersection area
  # inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
  inter_area = numpy.clamp(inter_rect_x2 - inter_rect_x1 + 1, 0) * numpy.clamp(inter_rect_y2 - inter_rect_y1 + 1, 0)

  #Union Area
  b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

  iou = inter_area / (b1_area + b2_area - inter_area)

  return iou


def process_results(prediction, confidence = 0.5, num_classes = 80, nms_conf = 0.4):
  prediction = prediction.detach().cpu().data
  conf_mask = (prediction[:,:,4] > confidence)
  conf_mask = np.expand_dims(conf_mask, 2)
  prediction = prediction * conf_mask
  
  # Non max suppression
  # box_corner = Tensor.uniform(tuple(prediction.shape))
  # box_corner = prediction.cpu().data
  box_corner = prediction
  box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
  box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
  box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
  box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
  prediction[:,:,:4] = box_corner[:,:,:4]

  batch_size = prediction.shape[0]

  write = False

  # Process img
  img_pred = prediction[0]

  """
  for i in range(batch_size):
    img_pred = prediction[i]
  """
  def numpy_max(input, dim):
    # Input -> tensor (10x8)
    return np.amax(input, axis=dim), np.argmax(input, axis=dim)
  
  # max_conf, max_conf_score = numpy.amax(image_pred[:,5:5+ num_classes], 1)
  max_conf, max_conf_score = numpy_max(img_pred[:,5:5 + num_classes], 1)
  # max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
  # max_conf = max_conf.float().unsqueeze(1)
  max_conf_score = np.expand_dims(max_conf_score, axis=1)
  # max_conf_score = max_conf_score.float().unsqueeze(1)
  max_conf = np.expand_dims(max_conf, axis=1)
  seq = (img_pred[:,:5], max_conf, max_conf_score)
  image_pred = np.concatenate(seq, axis=1)
  # image_pred = torch.cat(seq, 1)

  # non_zero_ind =  (torch.nonzero(image_pred[:,4]))

  non_zero_ind = np.nonzero(image_pred[:,4])[0] # TODO: Check if this is right
  print("non zero ind")
  print(non_zero_ind.shape)
  image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
  print("Image_pred_")
  print(image_pred_.shape)
  try:
    # image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
    image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
  except:
    print("No detections found!")
    pass
    # continue

  if image_pred_.shape[0] == 0:
    # continue
    print("Exception occurred!")
    pass
  
  def unique(tensor):
    # tensor_np = tensor.cpu().numpy()
    # tensor_np = tensor.cpu().data
    tensor_np = tensor
    unique_np = np.unique(tensor_np)
    return unique_np # Dunno if this even works
    unique_tensor = Tensor(unique_np)

    # tensor_res = Tensor(unique_np)
    # return tensor_res

  # Get the various classes detected in the image
  # img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index

  img_classes = unique(image_pred_[:, -1])

  for cls in img_classes:
    # perform NMS, get the detections with one particular class
    # cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
    print("Shapes:")
    print(image_pred_.shape)
    print((image_pred_[:,-1] == cls).shape)

    # This line makes no funckign sensee
    cls_mask = np.expand_dims(image_pred_*(image_pred_[:,-1] == cls), axis=1)
    class_mask_ind = np.nonzero(cls_mask[:,-2]).squeeze()
    # class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
    # image_pred_class = image_pred_[class_mask_ind].view(-1,7)
    image_pred_class = np.reshape(image_pred_[class_mask_ind], shape=(-1, 7))
    
    #sort the detections such that the entry with the maximum objectness
    #confidence is at the top
    # conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
    conf_sort_index = np.sort(image_pred_class[:,4], )
    image_pred_class = image_pred_class[conf_sort_index]
    idx = image_pred_class.size(0)   #Number of detections
    
    for i in range(idx):
        #Get the IOUs of all boxes that come after the one we are looking at 
        #in the loop
        try:
            ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
        except ValueError:
            break
    
        except IndexError:
            break
    
        #Zero out all the detections that have IoU > treshhold
        iou_mask = (ious < nms_conf).float().unsqueeze(1)
        image_pred_class[i+1:] *= iou_mask       
    
        #Remove the non-zero entries
        non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
        image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
        
    batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
    seq = batch_ind, image_pred_class
    
    if not write:
        output = torch.cat(seq,1)
        write = True
    else:
        out = torch.cat(seq,1)
        output = torch.cat((output,out))
  pass # TODO: Process prediciton


def imresize(img, w, h):
  return np.array(Image.fromarray(img).resize((w, h)))

def infer(model, img):
  img = np.array(img)
  # img = imresize(img, 608, 608)
  img = imresize(img, 416, 416)
  img = img[:,:,::-1].transpose((2,0,1))
  img = img[np.newaxis,:,:,:]/255.0

  prediction = model.forward(Tensor(img))
  return prediction


def parse_cfg(cfg):
  # Return a list of blocks
  #file = open(cfgfile, 'r')
  #lines = file.read().split('\n') # store the lines in a list
  lines = cfg.decode("utf-8").split('\n')
  # lines = cfg.split("\n")
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

# TODO: Speed up this function, avoid copying stuff from GPU to CPU
def predict_transform(prediction, inp_dim, anchors, num_classes):
  # batch_size = prediction.size(0)
  batch_size = prediction.shape[0]
  stride = inp_dim // prediction.shape[2]
  # stride =  inp_dim // prediction.size(2)
  grid_size = inp_dim // stride
  bbox_attrs = 5 + num_classes
  num_anchors = len(anchors)
  
  """
  prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
  prediction = prediction.transpose(1,2).contiguous()
  prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
  """
  prediction = prediction.reshape(shape=(batch_size, bbox_attrs*num_anchors, grid_size*grid_size))
  # Original PyTorch: transpose(1, 2) -> For some reason numpy.transpose order has to be reversed?
  # print(prediction.shape)
  # print(prediction.transpose(order=(2, 1)).reshape(shape=(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)).shape)
  # prediction = np.ascontiguousarray(prediction.transpose(order=(2, 1)))
  prediction = prediction.transpose(order=(2, 1))
  # print("Prediction:", prediction.shape)
  prediction = prediction.reshape(shape=(batch_size, grid_size*grid_size*num_anchors, bbox_attrs))


  anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  #Sigmoid the  centre_X, centre_Y. and object confidencce
  # TODO: Fix this
  # print(prediction.cpu().data[:,:,0])
  def dsigmoid(data):
    return 1/(1+np.exp(-data))
  prediction.cpu().data[:,:,0] = dsigmoid(prediction.cpu().data[:,:,0])
  prediction.cpu().data[:,:,1] = dsigmoid(prediction.cpu().data[:,:,1])
  prediction.cpu().data[:,:,4] = dsigmoid(prediction.cpu().data[:,:,4])
  # prediction[:,:,0] = (prediction[:,:,0].sigmoid())
  # prediction[:,:,1] = (prediction[:,:,1].sigmoid())
  # prediction[:,:,4] = (prediction[:,:,4].sigmoid())
  """
  prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
  prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
  prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
  """
  #Add the center offsets
  grid = np.arange(grid_size)
  a, b = np.meshgrid(grid, grid)

  # x_offset = a.reshape(shape=(-1, 1))
  # y_offset = b.reshape(shape=(-1, 1))
  x_offset = a.reshape((-1, 1))
  y_offset = b.reshape((-1, 1))
  # x_offset = torch.FloatTensor(a).view(-1,1)
  # y_offset = torch.FloatTensor(b).view(-1,1)

  """
  if CUDA:
    x_offset = x_offset.cuda()
    y_offset = y_offset.cuda()
  """

  # x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
  x_y_offset = np.concatenate((x_offset, y_offset), 1)
  x_y_offset = np.tile(x_y_offset, (1, num_anchors))
  x_y_offset = x_y_offset.reshape((-1,2))
  x_y_offset = np.expand_dims(x_y_offset, 0)

  prediction.cpu().data[:,:,:2] += x_y_offset
  # prediction[:,:,:2] += x_y_offset

  #log space transform height and the width
  # anchors = torch.FloatTensor(anchors)
  anchors = Tensor(anchors)

  """ TODO: This GPU Stuff
  if CUDA:
    anchors = anchors.cuda()
  """

  # anchors = anchors.cpu().data.tile((grid_size*grid_size, 1)).expand_dims(0) # .repeat(grid_size*grid_size, 1).unsqueeze(0)
  # anchors = anchors.cpu().data.tile((grid_size*grid_size, 1)).expand_dims(0)
  anchors = np.tile(anchors.cpu().data, (grid_size*grid_size, 1))
  anchors = np.expand_dims(anchors, 0)
  # prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
  prediction.cpu().data[:,:,2:4] = np.exp(prediction.cpu().data[:,:,2:4])*anchors

  prediction.cpu().data[:,:,5: 5 + num_classes] = dsigmoid((prediction.cpu().data[:,:, 5 : 5 + num_classes]))

  prediction.cpu().data[:,:,:4] *= stride

  return prediction


class Darknet:
  def __init__(self, cfg):
    self.blocks = parse_cfg(cfg)
    self.net_info, self.module_list = self.create_modules(self.blocks)
    # Inputs
    print("Modules length:", len(self.module_list))
    # print(self.module_list)
    """
    for i, module in enumerate(self.blocks[1:]):
      # Print module
      print(module["type"])
      # print(self.module_list[i][0])
      if self.module_list[i][0].weights is not None:
        print("Weights:", self.module_list[i][0].weights.shape)
    print("=== MODULE LIST END ===")
    """
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

        # print(f"{index}: Adding a Conv2d layer with filters: prev_filters: {prev_filters}, filters: {filters}")
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
          end = int(x["layers"][1])
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
  
  def dump_weights(self):
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      print(self.blocks[i + 1]["type"], "weights")
      if module_type == "convolutional":
        model = self.module_list[i]
        conv = model[0]
        print("Conv weights")
        print(conv.weights.cpu().data)
  
  def load_weights(self, url):
    weights = fetch(url)
    # fp = open(file, "rb")
    # First 5 values (major, minor, subversion, Images seen)
    # header = np.fromfile(fp, dtype=np.int32, count = 5)
    header = np.frombuffer(weights, dtype=np.int32, count = 5)
    self.seen = header[3]

    def numel(tensor):
      from functools import reduce
      return reduce(lambda x, y: x*y, tensor.shape)

    # weights = np.fromfile(fp, dtype=np.float32)
    weights = np.frombuffer(weights, dtype=np.float32)
    weights = weights[5:]

    ptr = 0
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      # print("loading weights for module_type " , module_type)
      if module_type == "convolutional":
        model = self.module_list[i]
        try: # we have batchnorm, load conv weights without biases, and batchnorm values
          batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
        except: # no batchnorm, load conv weights + biases
          batch_normalize = 0
        
        conv = model[0]

        if (batch_normalize):
          bn = model[1]

          # Get the number of weights of batchnorm
          # num_biases = bn.bias.shape
          num_bn_biases = numel(bn.bias)

          # Load weights
          # print("Loading BN biases", ptr, ":", ptr + num_bn_biases)
          bn_biases = Tensor(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

          # print("Loading BN weights", ptr, ":", ptr + num_bn_biases)
          bn_weights = Tensor(weights[ptr:ptr+num_bn_biases])
          ptr += num_bn_biases

          bn_running_mean = Tensor(weights[ptr:ptr+num_bn_biases])
          ptr += num_bn_biases

          bn_running_var = Tensor(weights[ptr:ptr+num_bn_biases])
          ptr += num_bn_biases

          # Cast the loaded weights into dims of model weights
          bn_biases = bn_biases.reshape(shape=tuple(bn.bias.shape))
          bn_weights = bn_weights.reshape(shape=tuple(bn.weight.shape))
          bn_running_mean = bn_running_mean.reshape(shape=tuple(bn.running_mean.shape))
          bn_running_var = bn_running_var.reshape(shape=tuple(bn.running_var.shape))

          # Copy data
          bn.bias = bn_biases # Idk if this works
          bn.weight = bn_weights
          bn.running_mean = bn_running_mean
          bn.running_var = bn_running_var
        else:
          # load biases of the conv layer
          num_biases = numel(conv.biases)

          # Load wieghts
          # print("Loading CONV biases", ptr, ":", ptr + num_biases)
          conv_biases = Tensor(weights[ptr: ptr+num_biases])
          ptr += num_biases

          # Reshape
          conv_biases = conv_biases.reshape(shape=tuple(conv.biases.shape))

          # Copy
          conv.biases = conv_biases
        
        # Load weighys for conv layers
        num_weights = numel(conv.weights)

        # print("Loading CONV weights", ptr, ":", ptr + num_weights)
        conv_weights = Tensor(weights[ptr:ptr+num_weights])
        ptr += num_weights

        conv_weights = conv_weights.reshape(shape=tuple(conv.weights.shape))
        conv.weights = conv_weights



  
  def forward(self, x):
    modules = self.blocks[1:]
    outputs = {} # Cached outputs for route layer
    write = 0

    for i, module in enumerate(modules):
      module_type = (module["type"])
      # print("Running through layer " + module_type)
      # print("Input shape:", x.shape)
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
          if (layers[1]) > 0: layers[1] = layers[1] - i
          
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]

          # x = np.concatenate((map1, map2), 1)
          x = Tensor(np.concatenate((map1.cpu().data, map2.cpu().data), 1))
      
      elif module_type == "shortcut":
        from_ = int(module["from"])
        """ Shape mismatch: (1, 64, 410, 410) vs. (1, 64, 412, 412)
        TODO: Implement shortcut
        print("Outputs:")
        print(outputs[i-1].shape)
        print(outputs[i+from_].shape)
        """
        x = outputs[i - 1] + outputs[i + from_]
      
      elif module_type == "yolo":
        anchors = self.module_list[i][0].anchors
        inp_dim = int(self.net_info["height"])
        num_classes = int(module["classes"])

        # Transform
        x = predict_transform(x, inp_dim, anchors, num_classes)
        if not write:
          detections = x
          write = 1
        else:
          # detections = np.concatenate((detections, x), 1)
          detections = Tensor(np.concatenate((detections.cpu().data, x.cpu().data), 1))
      
      outputs[i] = x
    
    return detections # Return detections


if __name__ == "__main__":
  # cfg = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')
  cfg = fetch('https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/cfg/yolov3.cfg')

  # Make deterministic
  np.random.seed(1337)

  # Start model
  model = Darknet(cfg)

  print("Loading weights file (237MB). This might take a while…")
  model.load_weights('https://pjreddie.com/media/files/yolov3.weights')

  if GPU:
    params = get_parameters(model)
    [x.gpu_() for x in params]

  # from PIL import Image
  # url = sys.argv[1]
  url = "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png"
  # url = "https://i.redd.it/rflitbaldl751.jpg"
  # url = "https://www.telegraph.co.uk/content/dam/cars/2016/04/11/Dashcam1_trans_NvBQzQNjv4BqPItlErHJmT3AsVLfg-otf_grG63UgcgwjHsyPCDdu4E.png"
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
  print("Prediction result:")
  print(prediction.shape)
  print(prediction.cpu().data[0][0][5:10])
  #print("Prediction:")
  # prediction = Tensor.ones(1, 27612, 85)
  # print(prediction)
  # prediction = Tensor.uniform(1, 27612, 85)
  prediction = process_results(prediction)
  # prediction = temp_process_results(prediction)
