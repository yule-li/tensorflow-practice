## Introduction to Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are responsible for the latest major breakthroughs in image recognition in the past few years.

### CNNS structure
CNNs generally follow a structure.  The main convolutional setup is (input array) -> **(convolutional filter layer) -> (Pooling) -> (Activation layer)**.  

### Convolution layer
In mathematics, a convolution is a function which is applied over the output of another function.  In our case, we will consider applying a matrix mutliplication (filter) across an image. See the below diagram for an example of how this may work.

![Convolutional Filter](./images/01_intro_cnn.png)

### Pooling layer
After this convolutional filter, it is common to apply a pooling layer.  This pooling may be a max-pooling or an average pooling or another aggregation.  One of the key concepts here is that the pooling layer has no parameters- while decreasing the layer size.  See the below diagram for an example of max-pooling.

![Convolutional Filter](./images/01_intro_cnn2.png)

### Activation layer
After the max pooling, there is generally an activation layer.  One of the more common activation layers is the ReLU (Rectified Linear Unit).  
