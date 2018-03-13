import tensorflow as tf
import pylab as plt

def main():
    batch_size = 128
    num_threads = 16
    data_path = 'cifar-data/'

    filename_list = [data_path + 'data_batch_{}.bin'.format(i+1) for i in range(5)]
    file_q = tf.train.string_input_producer(filename_list)
    image, label = read_data(file_q)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + (num_threads + 1)*batch_size

    image_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size,capacity, min_after_dequeue, num_threads = num_threads)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        '''
        for i in range(5):
            image_np, label_np = sess.run([image,label])
            plt.imshow(image_np)
            plt.show()
        '''
        imgs_np, labels_np = sess.run([image_batch,label_batch])
        print imgs_np.shape,labels_np.shape

        coord.request_stop()
        coord.join(threads)

def read_data(file_q):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_q)

    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    depth_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),[result.depth,result.height,result.width])
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    reshaped_image = tf.cast(result.uint8image,tf.float32)
    height = 24
    width = 24

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,height,width)
    result.label.set_shape([1])


    return result.uint8image, result.label

if __name__ == '__main__':
    main()
