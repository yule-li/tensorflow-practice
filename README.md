# Learning tensorflow by just one example for beginners.
This project was designed for learning tensorflow by just one example for beginners. More specifically, it's aimmed to achive the goal as follows:
- Understanding the key concepts of addressing compuation in tensorflow 
- Implementing your machine learning, especially deep learning, model using this concepts. 
- Learn to debug the problem and optimize the program under the tensorflow framework.

This project implemented a face recongnition model, actually face verification model. The model was trained on the CASIA-WEBFACE and tested on the LFW.

First the key concepts of tensorflow programming and components that constitude the model was introduced. Then a basic model taking advantages of both was implemented. Finally, the optimization was made to improve the basic model as far as both training speed and test accuracy were concerned.

## Table of Contents
0. [Introduction](http://study.163.com/course/courseLearn.htm?courseId=1005023019#/learn/video?lessonId=1051308829&courseId=1005023019)
    - Overview
    - Requirements

1. The key concepts
    - [Graph](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/graph/graph_add.ipynb)
    - [Session](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/session/session_add.ipynb)
    - [Tensor](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/tensor/tensors.ipynb)
    - [Operation](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/operations/basic_operations.ipynb)
2. Components
    - [Variables](https://github.com/yule-li/tensorflow-practice/blob/master/components/variables/variable.ipynb)
    - [Name and scope](https://github.com/yule-li/tensorflow-practice/blob/master/components/scopes/scopes.ipynb)
    - [Optimizer and trainer](https://github.com/yule-li/tensorflow-practice/blob/master/components/optimizer/linear_regression.ipynb)
    - [Convolution network](https://github.com/yule-li/tensorflow-networks/blob/master/networks/sphere_network.py)
    - Save and restore
    - Tensorboard
    - Customized layer
3. Work togother
    - Modularization
    - Project template
    - Code togother
4. Optimization
    - Speed:
        * Timeline
        * Data management
        * Multi-gpu
    - accuracy
        * Resnet
        * CocoFace

## References
1. [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
2. [Effective Tensorflow](https://github.com/vahidk/EffectiveTensorflow)
3. [Tensorflow sphereface](https://github.com/hujun100/tensorflow-sphereface)
