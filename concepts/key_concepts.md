Tensorflow program represent the compuation in data flow graph, which includes the main four concepts:
- graph
- session
- tensor
- operation

Tensorflow program can be divided into two phase:
- Assemble a graph(phase 1): you can model your problem by the graph
- Use a session to execute operations in the graph(phase 2)

It's important to be aware of which phase you are in. It can help you not only to do the right thing in different phase but also to understand the problem in tensroflow program.
