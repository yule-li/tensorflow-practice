import tensorflow as tf
import numpy as np
import tensorflow.contrib.data as tf_data
def data_with_one_shot():
    x = np.arange(0,10)

    #create the dataset from numpy array
    dx = tf_data.Dataset.from_tensor_slices(x)
    #dx = dx.repeat(2)

    #create a one-shot iterator
    iterator = dx.make_one_shot_iterator()
    #extracte an element
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for i in range(11):
            val = sess.run(next_element)
            print(val)
def data_with_init_iterator():
    x = np.arange(0,10)

    #create the dataset from numpy array
    dx = tf_data.Dataset.from_tensor_slices(x)
    #dx = dx.repeat(2)

    #create a one-shot iterator
    iterator = dx.make_initializable_iterator()
    #extracte an element
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(17):
            val = sess.run(next_element)
            print(val)
            if i % 9 == 0 and i > 0:
                sess.run(iterator.initializer)




def main():
    #data_with_one_shot()
    data_with_init_iterator()
if __name__ == '__main__':
    main()
