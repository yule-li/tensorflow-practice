import numpy as np
import sys
sys.path.insert(0,'lib')
import utils

class DataGenerator:
    def __init__(self,image_list,label_list,batch_size):
        self.image_list = image_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.dataset_size = len(self.image_list)
        self.indices = range(self.dataset_size)
        np.random.shuffle(self.indices)
        self.ind = 0
    def next_batch(self,image_height,image_width,do_random_crop=False,do_flip=True,do_prewhiten=True):
        if self.ind >= self.dataset_size:
            print 'ind and dataset_size',self.ind,self.dataset_size
            np.random.shuffle(self.indices)
            self.ind = 0
        true_num_batch = min(self.batch_size,self.dataset_size - self.ind)
        sample_paths = self.image_list[self.indices[self.ind:self.ind+true_num_batch]]
        sample_labels = self.label_list[self.indices[self.ind:self.ind+true_num_batch]]
        images = utils.load_data(sample_paths,do_random_crop,do_flip,image_height,image_width,do_prewhiten)
        self.ind += true_num_batch
        return images,sample_labels

