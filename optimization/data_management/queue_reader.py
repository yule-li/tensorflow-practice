import tensorflow as tf
def read_images_from_disk(input_queue,input_size):
    img_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img.set_shape((input_size[0],input_size[1],3))
    
    #img = tf.image.decode_image(img_contents, channels=3)
    print img
    label = input_queue[1]
    print label
    return img, label
    
class QueueReader(object):
    def __init__(self,image_list,label_list,input_size):
        self.images = tf.convert_to_tensor(image_list,dtype=tf.string)
        self.labels = tf.convert_to_tensor(label_list,dtype=tf.int32)
        self.input_size = input_size
        self.queue = tf.train.slice_input_producer([self.images,self.labels],shuffle=True)
        self.image, self.label = read_images_from_disk(self.queue,self.input_size)

    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements,num_threads=4)
        return image_batch, label_batch
        
