.. Tinygrad documentation master file, created by
   sphinx-quickstart on Tue Dec  5 11:21:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tinygrad's documentation!
====================================

Here you will find documentation for tinygrad, as well as some examples and tutorials.

Despite being a tiny library, tinygrad is capable of doing a lot of things. From state-of-the-art [vision](https://arxiv.org/abs/1905.11946) to state-of-the-art [language](https://arxiv.org/abs/1706.03762) models.


Vision
======

EfficientNet
------------

You can either pass in the URL of a picture to discover what it is:

.. code-block:: sh

    python3 examples/efficientnet.py https://media.istockphoto.com/photos/hen-picture-id831791190

Or, if you have a camera and OpenCV installed, you can detect what is in front of you:

.. code-block:: sh

    python3 examples/efficientnet.py webcam

YOLOv8
------

Take a look at :file:`yolov8.py`:

.. image:: /../showcase/yolov8_showcase_image.png
    :alt: yolov8 by tinygrad

Audio
=====

Whisper
-------

Take a look at :file:`whisper.py`. You need pyaudio and torchaudio installed.

.. code-block:: sh

    SMALL=1 python3 examples/whisper.py

Generative
==========

Generative Adversarial Networks
-------------------------------

Take a look at :file:`mnist_gan.py`:

.. image:: /../showcase/mnist_by_tinygrad.jpg
    :alt: mnist gan by tinygrad

Stable Diffusion
----------------

.. code-block:: sh

    python3 examples/stable_diffusion.py

.. image:: /../showcase/stable_diffusion_by_tinygrad.jpg
    :alt: a horse-sized cat eating a bagel

*"a horse-sized cat eating a bagel"*

LLaMA
-----

You will need to download and put the weights into the `weights/LLaMA` directory, which may need to be created.

Then you can have a chat with Stacy:

.. code-block:: sh

    python3 examples/llama.py




Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart
   notebooks/quickstart.ipynb
   notebooks/models.ipynb
   notebooks/JIT.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API reference

   api
   api_design



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`