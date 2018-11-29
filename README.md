# Food Image Recognition
This respository is used for storing code that I used/implemented for my senior thesis. The full research report can be viewed [HERE](http://www.henrypan.com/other/Final_Report_HanxiangPan.pdf).

## Dependencies
* All code are in [Matlab](http://www.mathworks.com/products/matlab/)
* Must manually download [GoogLeNet CNN](http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat) and [VGG-F CNN](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat) and put them into the **CNN** folder due to Github storage limit.
* [MatConvNet](http://www.vlfeat.org/matconvnet/quick/) - follow the instructions to install
* [Library](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) for Support Vector Machine
* The actual images used for training and testing are not included. Please check the directory path in each matlab file before running the scripts.

## Overview
**Utility Functions**
*  imageSetConstructor
*  confusionMatrix

**Classifiers**
*  K-nearest neighbors
*  Linear Regression
*  Perceptron Batch
*  Support Vector Machine


**Feature Extractors:**
*  Pixels
*  Edge Detectors (Prewitt, Sobel, Canny)
*  ConvNet (GoogLeNet, VGG-F)


