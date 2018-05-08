# Learning tensorflow by just one example for beginners.
This project was designed for learning tensorflow by just one example for beginners. More specifically, it's aimmed to achive the goal as follows:
- Understanding the key concepts of addressing compuation in tensorflow 
- Implementing your machine learning, especially deep learning, model using this concepts. 
- Learn to debug the problem and optimize the program under the tensorflow framework.

In order to achive this goal, I play attention on a pratical task and dig deep it instead of a wide of examples. By this way, I want to share the process of doing my research or task, and hope to help beginners to build a overview of modeling the machine learning problem using tensorflow.

This project choice face recongnition problem as the task, actually face verification model. The model was trained on the [CASIA-WEBFACE](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and tested on the [LFW](http://vis-www.cs.umass.edu/lfw/). More details about the dataset are [here](doc/dataset.md).

The project can be logically divided into three parts of different levels.
- the key concepts of tensorflow program and components that constitude the model were introduced. 
- a basic model taking advantages of both the key concepts and compoents was implemented and the softmax loss was used to learn the feature. 
- the optimization was made to improve the basic model as far as both training speed and test accuracy were concerned.
    * data management and multi-gpu were utilized to speed up the training
    * a more powerful network and new loss called [Large Margin Cosine Loss](https://arxiv.org/pdf/1801.09414.pdf) were used to improve the accuracy.

## Table of Contents
0. [Introduction](http://study.163.com/course/courseLearn.htm?courseId=1005023019#/learn/video?lessonId=1051308829&courseId=1005023019)
    - [Requirements](https://github.com/yule-li/tensorflow-practice/blob/master/introduction/requirement.md)
    - [Overview](https://github.com/yule-li/tensorflow-practice/blob/master/introduction/overview.md)
    - [Installation](https://github.com/yule-li/tensorflow-practice/blob/master/introduction/installation.md)
    - [Resources](https://github.com/yule-li/tensorflow-practice/blob/master/introduction/document.md)

1. The key concepts
    - [Graph](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/graph/graph_add.ipynb)
    - [Session](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/session/session_add.ipynb)
    - [Tensor](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/tensor/tensors.ipynb)
    - [Operation](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/operations/basic_operations.ipynb)
2. Components
    - [Variables](https://github.com/yule-li/tensorflow-practice/blob/master/components/variables/variable.ipynb)
    - [Name and scope](https://github.com/yule-li/tensorflow-practice/blob/master/components/scopes/scopes.ipynb)
    - [Optimizer and trainer](https://github.com/yule-li/tensorflow-practice/blob/master/components/optimizer/linear_regression.ipynb)
    - [Convolution network](https://github.com/yule-li/tensorflow-practice/tree/master/components/convolution_network)
    - [Save and restore](components/save_and_restore/save_and_restore.ipynb)
    - [Tensorboard]()
    - [Customized layer](components/customized_op/customized_op.ipynb)
3. Work togother
    - [Modularization](work_togother/modularization/modularization.ipynb)
    - [Project template](work_togother/https://github.com/MrGemy95/Tensorflow-Project-Template/tree/998f39bf2786980e3e3b171e9796148a3ec3322f)
    - [Code togother](https://github.com/yule-li/tensorflow-practice/blob/master/work_togother/tf_face/tf_face.ipynb)
4. Optimization
    - Speed:
        * [Timeline](optimization/timeline/timeline.ipynb)
        * [Data management](optimization/data_management/data_management.ipynb)
        * [Multi-gpu](optimization/multi_gpu/multi_gpus.ipynb)
    - accuracy
        * [Modify the network structure](optimization/inception_resnet_v1/inception_resnet_v1.ipynb)
        * [CosFace](https://github.com/yule-li/CosFace)

## How to debug your tensorflow program
Firstly you should be clear which phase the errors are from.

## FAQ
Please look [here](doc/faq.md).

## References
1. [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
2. [Effective Tensorflow](https://github.com/vahidk/EffectiveTensorflow)
3. [Tensorflow sphereface](https://github.com/hujun100/tensorflow-sphereface)
4. [tensorflow cookbook](https://github.com/nfmcclure/tensorflow_cookbook)
5. [tensorflow sphereface](https://github.com/hujun100/tensorflow-sphereface)


