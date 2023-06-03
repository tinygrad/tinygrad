# tinygrad Showcase

Despite being a tiny library, tinygrad is capable of doing a lot of things. From state-of-the-art [vision](https://arxiv.org/abs/1905.11946) to state-of-the-art [language](https://arxiv.org/abs/1706.03762) models.

## Vision

### EfficientNet

You can either pass in the URL of a picture to discover what it is:
```sh
python3 examples/efficientnet.py https://media.istockphoto.com/photos/hen-picture-id831791190
```
Or, if you have a camera and OpenCV installed, you can detect what is in front of you:
```sh
python3 examples/efficientnet.py webcam
```

### YOLOv3

Take a look at [yolov3.py](/examples/yolov3.py).

![yolo by tinygrad](/docs/showcase/yolo_by_tinygrad.jpg)

## Generative

### Generative Adversarial Networks

Take a look at [mnist_gan.py](/examples/mnist_gan.py).

![mnist gan by tinygrad](/docs/showcase/mnist_by_tinygrad.jpg)

### Stable Diffusion

You will need to download the [weights](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) of Stable Diffusion and put it into the [weights/](/weights) directory.

```sh
python3 examples/stable_diffusion.py
```

![a horse sized cat eating a bagel](/docs/showcase/stable_diffusion_by_tinygrad.jpg)
*"a horse sized cat eating a bagel"*

### LLaMA

You will need to download and put the weights into the [weights/LLaMA](/weightsLLaMA) directory, which may need to be created.

Then you can have a chat with Stacy:
```sh
python3 examples/llama.py
```
