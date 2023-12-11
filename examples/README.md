# tinygrad Examples

Welcome to the examples section of tinygrad! This collection showcases the versatility and power of tinygrad, a minimalist deep learning library.

| Example File                 | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `efficientnet.py`            | Demonstrates image classification using EfficientNet, both from URLs and live webcam input.     |
| `yolov8.py`                  | Showcases object detection capabilities using the YOLOv8 model.                                 |
| `whisper.py`                 | Implements speech recognition using the Whisper model, requires pyaudio and torchaudio.         |
| `mnist_gan.py`               | Introduces Generative Adversarial Networks (GANs) with the MNIST dataset.                       |
| `stable_diffusion.py`        | Generates images from text descriptions using the Stable Diffusion model.                       |
| `llama.py`                   | Interacts with the LLaMA model for various natural language processing tasks.                   |

Each of these examples offers a unique insight into the capabilities of the tinygrad library, demonstrating its application in various domains such as vision, audio, and generative tasks. For more detailed information on each example, including usage instructions. 


## Examples of how to use them

### EfficientNet
- **URL Image Classification:** Classify images from URLs.
  ```bash
  python3 examples/efficientnet.py [URL]
  ```
- **Webcam Image Classification:** Detect objects using your webcam.
  ```bash
  python3 examples/efficientnet.py webcam
  ```

### Whisper
- Experience speech recognition with Whisper.
  ```bash
  SMALL=1 python3 examples/whisper.py
  ```

### Stable Diffusion
- Generate creative images with text descriptions.
  ```bash
  python3 examples/stable_diffusion.py
  ```

### LLaMA
- Interact with the LLaMA model for natural language processing.
  ```bash
  python3 examples/llama.py
  ```
