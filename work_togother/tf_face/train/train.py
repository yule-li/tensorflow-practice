
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import os
import time
import sys
sys.path.insert(0,'lib')
sys.path.insert(0,'networks')
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
import tensorflow.contrib.data as tf_data
from collections import Counter
import numpy as np
import importlib
import itertools
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
import argparse
import utils
import sphere_network as network
from data_generator import DataGenerator

#import lfw
import pdb
#import cv2
#import pylab as plt

debug = False
softmax_ind = 0

from tensorflow.python.ops import data_flow_ops

def _from_tensor_slices(tensors_x,tensors_y):
    #return TensorSliceDataset((tensors_x,tensors_y))
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))



def main(args):
  
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    utils.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)

    train_set = utils.get_dataset(args.data_dir)
    nrof_classes = len(train_set)
    print('nrof_classes: ',nrof_classes)
    image_list, label_list = utils.get_image_paths_and_labels(train_set)
    image_list = np.array(image_list)
    print('total images: {}'.format(len(image_list)))
    label_list = np.array(label_list,dtype=np.int32)

    dataset_size = len(image_list)
    data_reader = DataGenerator(image_list,label_list,args.batch_size)
    

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
        
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False,name='global_step')

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        images_placeholder = tf.placeholder(tf.float32,[None,112,96,3], name='images_placeholder')
        labels_placeholder = tf.placeholder(tf.int32,[None], name='labels_placeholder')
        
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
                
        
        
            

    
        
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        print('Using optimizer: {}'.format(args.optimizer))
        if args.optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate,0.9)
                        
        if args.network ==  'sphere_network':
            prelogits = network.infer(images_placeholder)
        else:
            raise Exception('Not supported network: {}'.format(args.loss_type))
        
        if args.loss_type == 'softmax':
            cross_entropy_mean = utils.softmax_loss(prelogits,labels_placeholder, len(train_set),args.weight_decay,False)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #loss = cross_entropy_mean + args.weight_decay*tf.add_n(regularization_losses)
            loss = cross_entropy_mean + args.weight_decay*tf.add_n(regularization_losses)
            #loss = cross_entropy_mean 
        else:
            raise Exception('Not supported loss type: {}'.format(args.loss_type))
       
        #loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        losses = {}
        losses['total_loss'] = loss
        losses['softmax_loss'] = cross_entropy_mean
        debug_info = {}
        debug_info ['prelogits'] = prelogits
        

        grads = opt.compute_gradients(loss,tf.trainable_variables())
        train_op = opt.apply_gradients(grads,global_step=global_step)

        #save_vars = [var for var in tf.global_variables() if 'Adagrad' not in var.name and 'global_step' not in var.name]
        save_vars = tf.global_variables() 
         
        #saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        saver = tf.train.Saver(save_vars, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})


        
        with sess.as_default():
            #pdb.set_trace()

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
        
                # Train for one epoch
                train(args, sess, epoch, images_placeholder, labels_placeholder, data_reader,debug,
                     learning_rate_placeholder,  global_step, 
                     losses, train_op, args.learning_rate_schedule_file)

                # Save variables and the metagraph if it doesn't exist already
                model_dir = args.models_base_dir
                checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % 'softmax')
                saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

                # Evaluate on LFW
    return model_dir

def train(args, sess, epoch, images_placeholder, labels_placeholder, data_reader,debug,
          learning_rate_placeholder,  global_step, 
          loss, train_op, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        start_time = time.time()
        
        print('Running forward pass on sampled images: ', end='')
        images, labels = data_reader.next_batch(args.image_height,args.image_width)
        #if epoch < 10:
        #    continue
        #if batch_number > 49:
        #    pdb.set_trace()
        #    print(labels)
        #print(images.shape,labels)
        feed_dict = {learning_rate_placeholder: lr ,images_placeholder:images, labels_placeholder:labels}
        start_time = time.time()
        total_err, softmax_err, _, step = sess.run([loss['total_loss'], loss['softmax_loss'], train_op, global_step ], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTotal Loss %2.3f\tSoftmax Loss %2.3f, lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, total_err, softmax_err, lr))

        batch_number += 1
    return step
 

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='logs/facenet_ms_mp')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='models/facenet_ms_mp')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=.9)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--loss_type', type=str,
        help='Which type loss to be used.',default='softmax')
    parser.add_argument('--network', type=str,
        help='which network is used to extract feature.',default='resnet50')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--image_src_size', type=int,
        help='Src Image size (height, width) in pixels.', default=256)
    parser.add_argument('--image_height', type=int,
        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_width', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=30)
    parser.add_argument('--num_gpus', type=int,
        help='Number of gpus.', default=4)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=5)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=600)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=256)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM','SGD'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
