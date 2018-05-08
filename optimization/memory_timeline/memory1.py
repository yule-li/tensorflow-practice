import tensorflow as tf
from tensorflow.python.client import timeline
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

if __name__ == '__main__':
    a = tf.constant([1,2],name='a')
    b = tf.constant(3,name='b')
    c = tf.add(a,b,name='c')
    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    
    config = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L0)))
    with tf.Session(config=config) as sess:
        c_np = sess.run(c,options=run_options,run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()

        with open('tl.json','w') as wd:
            wd.write(ctf)
