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
from yolo_nn import Conv2d, Upsample, EmptyLayer, DetectionLayer, LeakyReLU, MaxPool2d
from tinygrad.nn import BatchNorm2D

import cv2
from PIL import Image

def show_labels(prediction, confidence = 0.5, num_classes = 80):
  coco_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
  coco_labels = coco_labels.decode('utf-8').split('\n')

  prediction = prediction.detach().cpu().data

  conf_mask = (prediction[:,:,4] > confidence)
  conf_mask = np.expand_dims(conf_mask, 2)
  prediction = prediction * conf_mask

  def numpy_max(input, dim):
    # Input -> tensor (10x8)
    return np.amax(input, axis=dim), np.argmax(input, axis=dim)
  
  # Iterate over batches
  for i in range(prediction.shape[0]):
    img_pred = prediction[i]
    max_conf, max_conf_score = numpy_max(img_pred[:,5:5 + num_classes], 1)
    max_conf_score = np.expand_dims(max_conf_score, axis=1)
    max_conf = np.expand_dims(max_conf, axis=1)
    seq = (img_pred[:,:5], max_conf, max_conf_score)
    image_pred = np.concatenate(seq, axis=1)

    non_zero_ind = np.nonzero(image_pred[:,4])[0]
    assert(all(image_pred[non_zero_ind,0] > 0))

    image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
    try:
      image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
    except:
      print("No detections found!")
      pass
    classes, indexes = np.unique(image_pred_[:, -1], return_index=True)
    for index, coco_class in enumerate(classes):
      probability = image_pred_[indexes[index]][4] * 100
      print("Detected", coco_labels[int(coco_class)], "{:.2f}%".format(probability))

def letterbox_image(img, inp_dim=608):
  img_w, img_h = img.shape[1], img.shape[0]
  w, h = inp_dim
  new_w = int(img_w * min(w/img_w, h/img_h))
  new_h = int(img_h * min(w/img_w, h/img_h))
  resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
  
  canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
  canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
  
  return canvas

def add_boxes(img, prediction):
  if isinstance(prediction, int): # no predictions
    return img
  coco_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
  coco_labels = coco_labels.decode('utf-8').split('\n')
  height, width = img.shape[0:2]
  scale_factor = 608 / width

  prediction[:,[1,3]] -= (608 - scale_factor * width) / 2
  prediction[:,[2,4]] -= (608 - scale_factor * height) / 2

  for i in range(prediction.shape[0]):
    pred = prediction[i]
    corner1 = tuple(pred[1:3].astype(int))
    corner2 = tuple(pred[3:5].astype(int))
    w = corner2[0] - corner1[0]
    h = corner2[1] - corner1[1]
    corner2 = (corner2[0] + w, corner2[1] + h)
    label = coco_labels[int(pred[-1])]
    img = cv2.rectangle(img, corner1, corner2, (255, 0, 0), 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = corner1[0] + t_size[0] + 3, corner1[1] + t_size[1] + 4
    img = cv2.rectangle(img, corner1, c2, (255, 0, 0), -1)
    img = cv2.putText(img, label, (corner1[0], corner1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
  
  return img

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
  inter_rect_x1 = np.maximum(b1_x1, b2_x1)
  inter_rect_y1 = np.maximum(b1_y1, b2_y1)
  inter_rect_x2 = np.maximum(b1_x2, b2_x2)
  inter_rect_y2 = np.maximum(b1_y2, b2_y2)

  #Intersection area
  inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, 99999) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, 99999)

  #Union Area
  b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

  iou = inter_area / (b1_area + b2_area - inter_area)

  return iou


def process_results(prediction, confidence = 0.9, num_classes = 80, nms_conf = 0.4):
  prediction = prediction.detach().cpu().data
  conf_mask = (prediction[:,:,4] > confidence)
  conf_mask = np.expand_dims(conf_mask, 2)
  prediction = prediction * conf_mask
  
  # Non max suppression
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

  def numpy_max(input, dim):
    # Input -> tensor (10x8)
    return np.amax(input, axis=dim), np.argmax(input, axis=dim)
  
  max_conf, max_conf_score = numpy_max(img_pred[:,5:5 + num_classes], 1)
  max_conf_score = np.expand_dims(max_conf_score, axis=1)
  max_conf = np.expand_dims(max_conf, axis=1)
  seq = (img_pred[:,:5], max_conf, max_conf_score)
  image_pred = np.concatenate(seq, axis=1)

  non_zero_ind = np.nonzero(image_pred[:,4])[0]
  assert(all(image_pred[non_zero_ind,0] > 0))
  image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
  try:
    image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
  except:
    print("No detections found!")
    return 0

  if image_pred_.shape[0] == 0:
    print("No detections found!")
    return 0
  
  def unique(tensor):
    tensor_np = tensor
    unique_np = np.unique(tensor_np)
    return unique_np

  img_classes = unique(image_pred_[:, -1])

  for cls in img_classes:
    # perform NMS, get the detections with one particular class
    cls_mask = image_pred_*np.expand_dims(image_pred_[:, -1] == cls, axis=1)
    class_mask_ind = np.squeeze(np.nonzero(cls_mask[:,-2]))
    # class_mask_ind = np.nonzero()
    image_pred_class = np.reshape(image_pred_[class_mask_ind], (-1, 7))
    
    # sort the detections such that the entry with the maximum objectness
    # confidence is at the top
    conf_sort_index = np.argsort(image_pred_class[:,4])
    image_pred_class = image_pred_class[conf_sort_index]
    idx = image_pred_class.shape[0]   #Number of detections
    
    for i in range(idx):
      #Get the IOUs of all boxes that come after the one we are looking at 
      #in the loop
      try:
        ious = bbox_iou(np.expand_dims(image_pred_class[i], axis=0), image_pred_class[i+1:])
      except ValueError:
        break
  
      except IndexError:
        break
  
      # Zero out all the detections that have IoU > treshhold
      iou_mask = np.expand_dims((ious < nms_conf), axis=1)
      image_pred_class[i+1:] *= iou_mask
  
      # Remove the non-zero entries
      non_zero_ind = np.squeeze(np.nonzero(image_pred_class[:,4]))
      image_pred_class = np.reshape(image_pred_class[non_zero_ind], (-1, 7))    

    batch_ind = np.array([[0]])
    seq = (batch_ind, image_pred_class)
    
    if not write:
      output = np.concatenate(seq, 1)
      write = True
    else:
      out = np.concatenate(seq, axis=1)
      output = np.concatenate((output,out))
  try:
    return output
  except:
    return 0

def imresize(img, w, h):
  return np.array(Image.fromarray(img).resize((w, h)))

def resize(img, inp_dim=(608, 608)):
  img_w, img_h = img.shape[1], img.shape[0]
  w, h = inp_dim
  new_w = int(img_w * min(w/img_w, h/img_h))
  new_h = int(img_h * min(w/img_w, h/img_h))
  resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
  
  canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
  canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
  
  return canvas

def infer(model, img):
  img = np.array(img)
  img = imresize(img, 608, 608)
  # img = resize(img)
  img = img[:,:,::-1].transpose((2,0,1))
  img = img[np.newaxis,:,:,:]/255.0

  prediction = model.forward(Tensor(img))
  return prediction


def parse_cfg(cfg):
  # Return a list of blocks
  lines = cfg.decode("utf-8").split('\n')
  lines = [x for x in lines if len(x) > 0]
  lines = [x for x in lines if x[0] != '#']
  lines = [x.rstrip().lstrip() for x in lines]

  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[":
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block["type"] = line[1:-1].rstrip()
    else:
      key,value = line.split("=")
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks

# TODO: Speed up this function, avoid copying stuff from GPU to CPU
def predict_transform(prediction, inp_dim, anchors, num_classes):
  batch_size = prediction.shape[0]
  stride = inp_dim // prediction.shape[2]
  grid_size = inp_dim // stride
  bbox_attrs = 5 + num_classes
  num_anchors = len(anchors)
  
  prediction = prediction.reshape(shape=(batch_size, bbox_attrs*num_anchors, grid_size*grid_size))
  # Original PyTorch: transpose(1, 2) -> For some reason numpy.transpose order has to be reversed?
  prediction = prediction.transpose(order=(0, 2, 1))
  prediction = prediction.reshape(shape=(batch_size, grid_size*grid_size*num_anchors, bbox_attrs))
  
  # st = time.time()
  prediction_cpu = prediction.cpu().data
  # print('put on CPU in %.2f s' % (time.time() - st))

  anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  #Sigmoid the  centre_X, centre_Y. and object confidence
  # TODO: Fix this
  def dsigmoid(data):
    return 1/(1+np.exp(-data))
  
  prediction_cpu[:,:,0] = dsigmoid(prediction_cpu[:,:,0])
  prediction_cpu[:,:,1] = dsigmoid(prediction_cpu[:,:,1])
  prediction_cpu[:,:,4] = dsigmoid(prediction_cpu[:,:,4])
  
  # Add the center offsets
  grid = np.arange(grid_size)
  a, b = np.meshgrid(grid, grid)

  x_offset = a.reshape((-1, 1))
  y_offset = b.reshape((-1, 1))

  x_y_offset = np.concatenate((x_offset, y_offset), 1)
  x_y_offset = np.tile(x_y_offset, (1, num_anchors))
  x_y_offset = x_y_offset.reshape((-1,2))
  x_y_offset = np.expand_dims(x_y_offset, 0)

  prediction_cpu[:,:,:2] += x_y_offset

  anchors = np.tile(anchors, (grid_size*grid_size, 1))
  anchors = np.expand_dims(anchors, 0)

  prediction_cpu[:,:,2:4] = np.exp(prediction_cpu[:,:,2:4])*anchors
  prediction_cpu[:,:,5: 5 + num_classes] = dsigmoid((prediction_cpu[:,:, 5 : 5 + num_classes]))
  prediction_cpu[:,:,:4] *= stride
  prediction.gpu_()

  return Tensor(prediction_cpu)


class Darknet:
  def __init__(self, cfg):
    self.blocks = parse_cfg(cfg)
    self.net_info, self.module_list = self.create_modules(self.blocks)
    print("Modules length:", len(self.module_list))

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
        
        conv = Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), pad, bias = bias)
        module.append(conv)

        # BatchNorm2d
        if batch_normalize:
          bn = BatchNorm2D(filters, eps=1e-05, training=True, track_running_stats=True)
          module.append(bn)

        # LeakyReLU activation
        if activation == "leaky":
          module.append(LeakyReLU(0.1))
      
      # TODO: Add tiny model
      elif module_type == "maxpool":
        size = int(x["size"])
        stride = int(x["stride"])
        maxpool = MaxPool2d(size, stride)
        module.append(maxpool)

      elif module_type == "upsample":
        upsample = Upsample(scale_factor = 2, mode = "nearest")
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
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
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
      if module_type == "convolutional":
        print(self.blocks[i + 1]["type"], "weights", i)
        model = self.module_list[i]
        conv = model[0]
        print(conv.weight.cpu().data[0][0][0])
        if conv.bias is not None:
          print("biases")
          print(conv.bias.shape)
          print(conv.bias.cpu().data[0][0:5])
        else:
          print("None biases for layer", i)
  
  def load_weights(self, url):
    weights = fetch(url)
    # First 5 values (major, minor, subversion, Images seen)
    header = np.frombuffer(weights, dtype=np.int32, count = 5)
    self.seen = header[3]

    def numel(tensor):
      from functools import reduce
      return reduce(lambda x, y: x*y, tensor.shape)

    weights = np.frombuffer(weights, dtype=np.float32)
    weights = weights[5:]

    ptr = 0
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]

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
          num_bn_biases = numel(bn.bias)

          # Load weights
          bn_biases = Tensor(weights[ptr:ptr + num_bn_biases])
          ptr += num_bn_biases

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
          bn.bias = bn_biases
          bn.weight = bn_weights
          bn.running_mean = bn_running_mean
          bn.running_var = bn_running_var
        else:
          # load biases of the conv layer
          num_biases = numel(conv.bias)

          # Load wieghts
          conv_biases = Tensor(weights[ptr: ptr+num_biases])
          ptr += num_biases

          # Reshape
          conv_biases = conv_biases.reshape(shape=tuple(conv.bias.shape))

          # Copy
          conv.bias = conv_biases
        
        # Load weighys for conv layers
        num_weights = numel(conv.weight)

        conv_weights = Tensor(weights[ptr:ptr+num_weights])
        ptr += num_weights

        conv_weights = conv_weights.reshape(shape=tuple(conv.weight.shape))
        conv.weight = conv_weights



  
  def forward(self, x):
    modules = self.blocks[1:]
    outputs = {} # Cached outputs for route layer
    write = 0

    for i, module in enumerate(modules):
      module_type = (module["type"])
      st = time.time()
      if module_type == "convolutional" or module_type == "upsample":
        for index, layer in enumerate(self.module_list[i]):
          x = layer(x)
      
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

          x = Tensor(np.concatenate((map1.cpu().data, map2.cpu().data), 1))
      
      elif module_type == "shortcut":
        from_ = int(module["from"])
        x = outputs[i - 1] + outputs[i + from_]
      
      elif module_type == "yolo":
        anchors = self.module_list[i][0].anchors
        inp_dim = int(self.net_info["height"])
        # inp_dim = 416

        num_classes = int(module["classes"])
        # Transform
        x = predict_transform(x, inp_dim, anchors, num_classes)
        if not write:
          detections = x
          write = 1
        else:
          detections = Tensor(np.concatenate((detections.cpu().data, x.cpu().data), 1))
      
      # print(module_type, 'layer took %.2f s' % (time.time() - st))
      outputs[i] = x
    
    return detections # Return detections

if __name__ == "__main__":
  cfg = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg') # normal model
  # cfg = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg') # tiny model

  # Make deterministic
  np.random.seed(1337)

  # Start model
  model = Darknet(cfg)

  print("Loading weights file (237MB). This might take a while…")
  model.load_weights('https://pjreddie.com/media/files/yolov3.weights') # normal model
  # model.load_weights('https://pjreddie.com/media/files/yolov3-tiny.weights') # tiny model

  if GPU:
    params = get_parameters(model)
    [x.gpu_() for x in params]

  if len(sys.argv) > 1:
    url = sys.argv[1]
  else:
    url = "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png"

  img = None
  # We use cv2 because for some reason, cv2 imread produces better results?
  if url == 'webcam':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])

      prediction = infer(model, img)
      prediction = process_results(prediction)

      boxes = add_boxes(imresize(np.array(img), 608, 608), prediction)
      boxes = cv2.cvtColor(boxes, cv2.COLOR_RGB2BGR)
      cv2.imshow('yolo', boxes)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
  elif url.startswith('http'):
    img_stream = io.BytesIO(fetch(url))
    img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
  else:
    img = cv2.imread(url)
  
  # Predict
  st = time.time()
  print('running inference…')
  prediction = infer(model, img)
  print('did inference in %.2f s' % (time.time() - st))

  labels = show_labels(prediction)
  prediction = process_results(prediction)
  # print(prediction)
  boxes = add_boxes(imresize(img, 608, 608), prediction)
  # Save img
  cv2.imwrite('boxes.jpg', boxes)
