---
layout:
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# Showcase

Despite being a tiny library, tinygrad is capable of doing a lot of things. From state-of-the-art [vision](https://arxiv.org/abs/1905.11946) to state-of-the-art [language](https://arxiv.org/abs/1706.03762) models.

### Vision

#### EfficientNet

You can either pass in the URL of a picture to discover what it is:

```bash
python3 examples/efficientnet.py https://media.istockphoto.com/photos/hen-picture-id831791190
```

Or, if you have a camera and OpenCV installed, you can detect what is in front of you:

```bash
python3 examples/efficientnet.py webcam
```

#### YOLOv8

Take a look at [yolov8.py](https://github.com/geohot/tinygrad/blob/master/examples/yolov8.py).

<figure><img src=".gitbook/assets/yolov8_showcase_image (1).png" alt=""><figcaption></figcaption></figure>

### Audio

#### Whisper

Take a look at [whisper.py](https://github.com/geohot/tinygrad/blob/master/examples/whisper.py). You need pyaudio and torchaudio installed.

```bash
SMALL=1 python3 examples/whisper.py
```

### Generative

#### Generative Adversarial Networks

Take a look at [mnist\_gan.py](https://github.com/geohot/tinygrad/blob/master/examples/mnist\_gan.py).

<figure><img src=".gitbook/assets/mnist_by_tinygrad.jpg" alt=""><figcaption></figcaption></figure>

#### Stable Diffusion

```
python3 examples/stable_diffusion.py
```

<figure><img src=".gitbook/assets/stable_diffusion_by_tinygrad.jpg" alt=""><figcaption><p><em>"a horse sized cat eating a bagel"</em></p></figcaption></figure>

#### LLaMA

You will need to download and put the weights into the `weights/LLaMA` directory, which may need to be created.

Then you can have a chat with Stacy:

```
python3 examples/llama.py
```
