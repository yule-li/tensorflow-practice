import sys
import time
sys.path.insert(0,'lib')
import utils
from queue_reader import QueueReader
import tensorflow as tf

def main():
    data_dir = 'dataset/CASIA-WebFace-112X96' 
    train_set = utils.get_dataset(data_dir)
    nrof_classes = len(train_set)
    print('nrof_classes: ',nrof_classes)
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    input_size = (112,96)#h,w
    qr = QueueReader(image_list,label_list,input_size)
    batch_size = 100
    images, labels = qr.dequeue(batch_size)
    sess =tf.Session()
    coord  = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    num_iter = 100
    for i in range(num_iter):
        t = time.time()
        np_imgs,np_labels = sess.run([images,labels])
        #print np_labels
        print('Load {} images cost time is {}'.format(np_imgs.shape[0],time.time()-t))


if __name__ == '__main__':
    main()
