## [What is TensorFlow](https://tensorflow.google.cn/)
TensorFLow is an open source software library for numerical computation using data flow **graphs**.

The overview of a similar graph of a three-layer neural network in TensorFlow shows below. Nodes in the graph represent mathermatical operations, while the graph edges represent the multidimensional data arrays(tensors) communicated between them.

**static graph vs dynamic graph**
- static graph: you define graph statically before a model can run. All communication with outer world is performed via ```tf.Session``` object and ```tf.Placeholder``` 
- dynamic graph: you can define, change and execute nodes as you go, no special session interfaces or placeholders.
![image](http://adventuresinmachinelearning.com/wp-content/uploads/2017/03/TensorFlow-data-flow-graph.gif)

## Why Tensorflow
- Flexibility
- Scalability
- Popularity

## [Data Flow Graphs](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/key_concepts.md)
Basic elements:
- graph
- session
- tensor
- operation

[Phases](https://github.com/yule-li/tensorflow-practice/blob/master/concepts/session/session_add.ipynb)
- Phase 1: assemble a graph
- Phase 2: use a session to execute operations in the graph

