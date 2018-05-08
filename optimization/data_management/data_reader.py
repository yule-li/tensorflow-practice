import tensorflow as tf
import tensorflow.contrib.data as tf_data
import sys
import numpy as np
import time
sys.path.insert(0,'lib')
import utils

softmax_ind = 0

def _from_tensor_slices(tensors_x,tensors_y):
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))
def main():
    data_dir = 'dataset/CASIA-WebFace-112X96' 
    train_set = utils.get_dataset(data_dir)
    nrof_classes = len(train_set)
    print('nrof_classes: ',nrof_classes)
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    image_list = np.array(image_list)
    label_list = np.array(label_list,dtype=np.int32)
    dataset_size = len(image_list)
    indices = range(dataset_size)
    np.random.shuffle(indices)

    batch_size = 100
    img_h = 112
    img_w = 96

    def _sample_people_softmax(x):
        global softmax_ind
        if softmax_ind >= dataset_size:
            np.random.shuffle(indices)
            softmax_ind = 0
        true_num_batch = min(batch_size,dataset_size - softmax_ind)

        sample_paths = image_list[indices[softmax_ind:softmax_ind+true_num_batch]]
        sample_labels = label_list[indices[softmax_ind:softmax_ind+true_num_batch]]

        softmax_ind += true_num_batch

        return (np.array(sample_paths), np.array(sample_labels,dtype=np.int32))

    def _parse_function(filename,label):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents, channels=3)
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        print(image.shape)
        return image, label

    epoch_size = 600
    max_nrof_epochs=10
    with tf.device("/cpu:0"):
        softmax_dataset = tf_data.Dataset.range(epoch_size*max_nrof_epochs*100)
        softmax_dataset = softmax_dataset.map(lambda x: tf.py_func(_sample_people_softmax,[x],[tf.string,tf.int32]))
        softmax_dataset = softmax_dataset.flat_map(_from_tensor_slices)
        softmax_dataset = softmax_dataset.map(_parse_function,num_threads=8,output_buffer_size=2000)
        softmax_dataset = softmax_dataset.batch(batch_size)
        softmax_iterator = softmax_dataset.make_initializable_iterator()
        softmax_next_element = softmax_iterator.get_next()
        softmax_next_element[0].set_shape((batch_size, img_h,img_w,3))
    with tf.Session() as sess:
        sess.run(softmax_iterator.initializer)
        for i in range(50):
            t = time.time()
            img_np,label_np = sess.run([softmax_next_element[0],softmax_next_element[1]])
            #print label_np
            print('Load {} images time cost: {}'.format(img_np.shape[0],time.time()-t))

if __name__ == '__main__':
    main()
