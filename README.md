# Masked Texture Synthesis with Multiresolution Pyramid Rendering

This repository contains a Tensorflow implementation which performs texture analysis given a doodle drawing and some input textures corresponding to the colors in this drawing. The algorithm will then create a composition using these input textures and the given drawing.

This algorithm makes use of Multiresolution Pyramid Rendering to improve the quality and increased resolution of the result.
To further improve the quality of the resulting composition we make use of several loss functions, including Gram, Histogram and Total Variance loss.

## Setup

#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [Jupyter Notebook](http://jupyter.org/) which can be installed through [Anaconda](https://www.continuum.io/DOWNLOADS)

#### Optional (but recommended) dependencies:
* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5+
* [cuDNN](https://developer.nvidia.com/cudnn) 5.0+

#### After installing the dependencies: 
* Download the [VGG-19 tensorflow weights](https://github.com/machrisaa/tensorflow-vgg)

## Usage
Run the iPython notebook `transfer.ipynb` and change the parameters. The code is sufficiently commented to explain itself.
