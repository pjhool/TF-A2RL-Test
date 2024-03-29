# TF-A2RL: Automatic Image Cropping
[[Project]](https://debangli.github.io/A2RL/)   [[Paper]](https://arxiv.org/abs/1709.04595)   [[Online Demo]](https://wuhuikai.github.io/TF-A2RL/)    [[API]](https://algorithmia.com/algorithms/wuhuikai/A2RL_online)   [[Related Work: GP-GAN (for Image Blending)]](https://github.com/wuhuikai/GP-GAN)

The official implementation for A2-RL: Aesthetics Aware Rinforcement Learning for Automatic Image Cropping

## Overview

| source | step 1 | step 2 | step 3 | step 4 | step 5 | output| 
| --- | --- | --- | --- | --- | --- | --- |
| ![](images/readme/source.png) | ![](images/readme/step1.png) | ![](images/readme/step2.png) | ![](images/readme/step3.png) | ![](images/readme/step4.png) | ![](images/readme/step5.png) | ![](images/readme/output.png) |

A2-RL (aka. Aesthetics Aware Reinforcement Learning) is the author's implementation of the RL-based automatic image cropping algorithm described in:
```
A2-RL: Aesthetics Aware Reinforcement Learning for Automatic Image Cropping   
Debang Li, Huikai Wu, Junge Zhang, Kaiqi Huang
```

Given a source image, our algorithm could take actions step by step to find almost the best cropping window on source image. 

Contact: Hui-Kai Wu (huikaiwu@icloud.com)

## View Finding Network
Evaluation
We provide the evaluation script to reproduce our evaluation results on Flickr cropping dataset. For example,

$ python vfn_eval.py --spp false --snapshot snapshots/model-wo-spp
You will need to get sliding_window.json and the test images from the Flickr cropping dataset and specify the path of your model when running vfn_eval.py. You can also try our pre-trained model, which can be downloaded from [[here]]( https://drive.google.com/drive/folders/0B0sDVRDPL5zBd3ozNlFmZEZpY1k).

Copy pre-trained model to View_Finding Direcotry 

If you want to get an aesthetic score of a patch, please take a look at the example featured by [[ModelDepot]]( https://modeldepot.io/yilingchen/view-finding-network ) 

## Getting started
* Install the python libraries. (See `Requirements`).
* Download the code from GitHub:
```bash
git clone https://github.com/pjhool/TF-A2RL-Test.git
cd TF-A2RL-Test
```
* Download the pretrained models `vfn_rl.pk` from [Google Drive](https://drive.google.com/open?id=0Bybnpq8dvwudREJnRWhFbk1rYW8), then put them in current directory (`TF-A2RL/`).

* Run the python script:
``` bash
python A2RL.py --image_path ./test_images --save_path ./test_images_cropped
```
or
``` bash
sh example.sh
```
* To A2RL Test for  AVA Landscape Images , download  [AVA Landscape Images ]( https://drive.google.com/open?id=1ojdHxuW8L10hjafwbMIJAAJsD6PYm8yU ) 
   


## Results compared with baseline methods (more [results](https://debangli.github.io/A2RL/))

|Source| VFN+Sliding window | A2-RL | Ground Truth |
| --- | --- | --- |---|
| ![](images/readme/1227.jpg) | ![](images/readme/vfn_1227.jpg) | ![](images/readme/a2rl_1227.jpg) | ![](images/readme/gt_1227.jpg) |
| ![](images/readme/1644.jpg) | ![](images/readme/vfn_1644.png) | ![](images/readme/output.png) | ![](images/readme/gt_1644.jpg) |
| ![](images/readme/2747.jpg) | ![](images/readme/vfn_2747.jpg) | ![](images/readme/a2rl_2747.jpg) | ![](images/readme/gt_2747.jpg) |
| ![](images/readme/2903.jpg) | ![](images/readme/vfn_2903.jpg) | ![](images/readme/a2rl_2903.jpg) | ![](images/readme/gt_2903.jpg) |
| ![](images/readme/9036.jpg) | ![](images/readme/vfn_9036.jpg) | ![](images/readme/a2rl_9036.jpg) | ![](images/readme/gt_9036.jpg) |

## Requirements
The code requires the following 3rd party libraries:
* pickle
* numpy
* [skimage](http://scikit-image.org/)
```bash
pip install scikit-image
```
Details see the official [README](https://github.com/scikit-image/scikit-image) for installing skimage.
* [TensorFlow](https://www.tensorflow.org/) 1.4 

Details see the official [README](https://github.com/tensorflow/tensorflow) for installing TensorFlow. 
## Command line arguments:
Type `python A2RL.py --help` for a complete list of the arguments.
* `--image_path`: path of the input image
* `--save_path`: path of output image
## Citation
```
@article{li2017a2,
  title={A2-RL: Aesthetics Aware Reinforcement Learning for Automatic Image Cropping},
  author={Li, Debang and Wu, Huikai and Zhang, Junge and Huang, Kaiqi},
  journal={arXiv preprint arXiv:1709.04595},
  year={2017}
}
```
