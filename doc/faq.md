# FAQ
___
Q1. Do not see anything  in browser when using tensorboard

A: You may change another browser to open it.

___
Q2. When I use ```tf.py_func``` to warp the python code into tensorflow operation, Do I need write the backward operation?

A: It depends on whether you want to use the gradient of this operation. If needed, you must implement the gradient of this operation and you can  choice the two method as follows:
    - override the gradient of ```tf.py_func``` using the tensorflow operation.
    - override the gradient of ```tf.py_func``` using the python code and you should warp the gradient computation by ```tf.py_func``` as tf operation as well.

