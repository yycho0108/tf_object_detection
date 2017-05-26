# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple transfer learning with an Inception v3 architecture model.

With support for TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.


To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

slim = tf.contrib.slim

from config import *

import signal

#from freeze_graph import freeze_graph

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
# pylint: enable=line-too-long

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 480
MODEL_INPUT_DEPTH = 3

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists(testing_percentage, validation_percentage):
  training_images = []
  testing_images = []
  validation_images = []

  for name in os.listdir(FLAGS.bottleneck_dir): # path = bottlenecks
    l = name.split('_lbl.npy')
    if len(l) <= 1:
      continue
    base_name = l[0] # without extensions, etc.
    hash_name = re.sub(r'_nohash_.*$', '', name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_IMAGES_PER_CLASS))
    if percentage_hash < validation_percentage:
      validation_images.append(base_name)
    elif percentage_hash < (testing_percentage + validation_percentage):
      testing_images.append(base_name)
    else:
      training_images.append(base_name)

  return {
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
      }

def report_graph(graph):
    for op in graph.get_operations():
        print('==' + op.name + '===')
        print('\t Input : ')
        for t in op.inputs:
            print('\t \t %s : %s' % (t.name, str(t.shape)))
        print('\t Output : ')
        for t in op.outputs:
            print('\t \t %s : %s' % (t.name, str(t.shape)))

def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.

    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    with tf.Graph().as_default() as graph:


        model_filename = os.path.join('./workspace/inception',
            'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            return_elements = list(OUTPUT_TENSOR_NAMES)
            return_elements += [JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]
            results = tf.import_graph_def(graph_def, name='', return_elements = return_elements)

            bottleneck_tensors = results[:-2]
            #print('bttt', bottleneck_tensors) # ===> list

            ## ADD A POOL ...
            for i in range(len(APPEND_POOL)):
                name = ('aux_pool_%d' % i)
                p = tf.nn.max_pool(bottleneck_tensors[-1],ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
                bottleneck_tensors.append(p)

            jpeg_data_tensor = results[-2]
            resized_input_tensor = results[-1]
                    
    #report_graph(graph)
    return graph, bottleneck_tensors, jpeg_data_tensor, resized_input_tensor

def get_bottleneck(sess, name, jpeg_data_tensor):
  btl_ext='_btl.npy'
  lbl_ext='_lbl.npy'

  bv_path = os.path.join(FLAGS.bottleneck_dir, name + btl_ext)
  bl_path = os.path.join(FLAGS.bottleneck_dir, name + lbl_ext)

  val = np.load(bv_path, allow_pickle=True)
  lab = np.load(bl_path, allow_pickle=True)
  return val, lab

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
    jpeg_data_tensor):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    # returned data = [batch_size, [num_clf, ...]]
    for _ in range(how_many):
      l = image_lists[category]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = l[image_index % len(l)]

      bottleneck, ground_truth = get_bottleneck(sess, image_name, jpeg_data_tensor)
      # [bottleneck] is a list of ndarrays
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)

      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for image_index, image_name in enumerate(image_lists[category]):
      bottleneck, ground_truth = get_bottleneck(sess, image_name, jpeg_data_tensor)

      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)

      filenames.append(image_name)

  formatted_btl = np.transpose(bottlenecks)
  formatted_btl = [np.stack(a,0) for a in formatted_btl]

  formatted_gt = np.transpose(ground_truths)
  formatted_gt = [np.stack(a,0) for a in formatted_gt]

  #formatted_btl = [np.stack([b[i] for b in bottlenecks], axis=0) for i in range(n)]
  #formatted_gt = [np.stack([g[i] for g in ground_truths], axis=0) for i in range(n)]
  return formatted_btl, formatted_gt, filenames

#def get_random_distorted_bottlenecks(
#    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
#    distorted_image, resized_input_tensor, bottleneck_tensor):
#  """Retrieves bottleneck values for training images, after distortions.
#
#  If we're training with distortions like crops, scales, or flips, we have to
#  recalculate the full model for every image, and so we can't use cached
#  bottleneck values. Instead we find random images for the requested category,
#  run them through the distortion graph, and then the full graph to get the
#  bottleneck results for each.
#
#  Args:
#    sess: Current TensorFlow Session.
#    image_lists: Dictionary of training images for each label.
#    how_many: The integer number of bottleneck values to return.
#    category: Name string of which set of images to fetch - training, testing,
#    or validation.
#    image_dir: Root folder string of the subfolders containing the training
#    images.
#    input_jpeg_tensor: The input layer we feed the image data to.
#    distorted_image: The output node of the distortion graph.
#    resized_input_tensor: The input node of the recognition graph.
#    bottleneck_tensor: The bottleneck output layer of the CNN graph.
#
#  Returns:
#    List of bottleneck arrays and their corresponding ground truths.
#  """
#  class_count = len(image_lists.keys())
#  bottlenecks = []
#  ground_truths = []
#  for unused_i in range(how_many):
#    label_index = random.randrange(class_count)
#    label_name = list(image_lists.keys())[label_index]
#    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
#    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
#                                category)
#    if not gfile.Exists(image_path):
#      tf.logging.fatal('File does not exist %s', image_path)
#    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
#    # Note that we materialize the distorted_image_data as a numpy array before
#    # sending running inference on the image. This involves 2 memory copies and
#    # might be optimized in other implementations.
#    distorted_image_data = sess.run(distorted_image,
#                                    {input_jpeg_tensor: jpeg_data})
#    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
#                                         resized_input_tensor,
#                                         bottleneck_tensor)
#    ground_truth = np.zeros(class_count, dtype=np.float32)
#    ground_truth[label_index] = 1.0
#    bottlenecks.append(bottleneck)
#    ground_truths.append(ground_truth)
#  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
  """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
  return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                            random_brightness):
    """Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.
      graph.

    Returns:
      The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                    MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_final_training_op(bottleneck_tensor):

    btl_shape = bottleneck_tensor.shape.as_list()

    n = btl_shape[1]
    m = btl_shape[2]
    k = btl_shape[3]

    with tf.name_scope('input'):
        shape = [None] + bottleneck_tensor.shape.dims
        bottleneck_input = tf.placeholder_with_default(
                bottleneck_tensor, shape=[None,n,m,k],
                name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32,
                [None, n, m, OUTPUT_DIMS],
                name='GroundTruthInput')
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            ## TODO : Deeper Layers?
            initial_value = tf.truncated_normal([btl_shape[-1], OUTPUT_DIMS], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([n, m, OUTPUT_DIMS]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            outputs = tf.tensordot(bottleneck_input, layer_weights, axes=[[3],[0]]) + layer_biases
            tf.summary.histogram('pre_activations', outputs)

        g_clf, g_bnd = tf.split(ground_truth_input, [NUM_CLASSES, 5*NUM_BOXES], 3, name='g_split') # ground truth
        n_clf, n_bnd = tf.split(outputs, [NUM_CLASSES, 5*NUM_BOXES], 3, name='n_split') # network output

    n_clf_s = tf.nn.softmax(n_clf, name='n_clf')
    tf.summary.histogram('activations', n_clf)

    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          labels=g_clf, logits=n_clf)
      with tf.name_scope('total'):
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('bbox_loss'):
        #g_bnd = (batch_size, n, m, 5*NUM_BOXES]
        bboxes = tf.split(g_bnd, NUM_BOXES, axis=3, name='bbox_split')
        for bbox in bboxes:
            iou = bbox[:,:,:,0]
            idx = tf.where(iou > 0.2) #TODO : Maybe This would work
            g = tf.gather_nd(g_bnd, idx) # l1 loss
            n = tf.gather_nd(n_bnd, idx)
            bbox_loss = tf.reduce_mean(tf.abs(g-n))

    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    tf.summary.scalar('bbox_loss', bbox_loss)
    loss = cross_entropy_mean + bbox_loss
    tf.summary.scalar('net_loss', loss)

    return (loss, cross_entropy_mean, bottleneck_input, ground_truth_input, n_clf_s, n_bnd)


def add_final_training_ops(bottleneck_tensors):
    net_loss = tf.Variable(0.0, name="net_loss")
    net_cross_entropy_mean = tf.Variable(0.0, name="net_cross_entropy")
    bottleneck_inputs = []
    ground_truth_inputs = []
    n_clfs = []
    n_bnds = []
    losses = []
    cross_entropy_means= []

    n = len(bottleneck_tensors)
    for i, bottleneck_tensor in enumerate(bottleneck_tensors):
        with tf.name_scope('btl_%d' % i):
            loss, cross_entropy_mean, bottleneck_input, ground_truth_input, n_clf, n_bnd = add_final_training_op(bottleneck_tensor)
            losses.append(loss)
            cross_entropy_means.append(cross_entropy_mean)
            bottleneck_inputs.append(bottleneck_input)
            ground_truth_inputs.append(ground_truth_input)
            n_clfs.append(n_clf)
            n_bnds.append(n_bnd)

    net_loss = tf.reduce_mean(losses)
    net_cross_entropy_mean = tf.reduce_mean(cross_entropy_means)

    with tf.name_scope('train'):
        ## TODO : replace optimizer?
        #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        train_step = optimizer.minimize(net_loss)

    return (train_step, net_cross_entropy_mean, bottleneck_inputs, ground_truth_inputs, n_clfs, n_bnds)

def decode_box(bnds):
    s_bnds = tf.split(bnds, NUM_BOXES, axis=3)

    res = []
    for i, bnd in enumerate(s_bnds):
        w_r, h_r = BBOX_RATIOS[i]
        n,m = bnd.shape.as_list()[1:3]
        iou, dx, dy, dw, dh = tf.unstack(bnd, axis=3)
        # convert to (iou, x, y, w, h)
        h, w = 299./n, 299./m # TODO : Warning : Hard Coded!

        i_, j_ = np.mgrid[0:n,0:m].astype(np.float32)
        i_ *= h # center positions
        i_ += (h/2.)
        j_ *= w
        j_ += (w/2.)
        x = dx * w + j_
        y = dy * h + i_
        w = dw * w * w_r
        h = dh * w * h_r

        res.append(tf.stack([iou, x, y, w, h], axis=3, name='decode_bbox')) # each iou has an associated prediction
    return tf.concat(res, axis=3, name='stack_box')


def add_evaluation_step(n_clf, n_bnd, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            #w_r,h_r = BBOX_RATIOS[idx]

            g_clf, g_bnd = tf.split(ground_truth_tensor, [NUM_CLASSES,5*NUM_BOXES], 3) # ground truth

            #n_clf, n_bnd = tf.split(final_tensor, [NUM_CLASSES,5*NUM_BOXES], 3) # ground truth

            #[batch,grid_i,grid_j,classes]
            g_clf_pred = tf.argmax(g_clf,axis=3) # class-prediction
            g_clf_val = tf.reduce_max(g_clf, axis=3)
            g_clf_mask = tf.greater(g_clf_val, 0.5) # = where object was identified
            g_clf_indices = tf.where(g_clf_mask)
            #g_iou_mask = tf.greater(g_bnd[:,:,:,0], 0.2) # = where bbox was identified 
            #g_bbox_indices = tf.where(tf.not_equal(tf.logical_and(g_clf_mask, g_iou_mask), 0))

            n_clf_pred = tf.argmax(n_clf,axis=3) # class-prediction
            n_clf_val = tf.reduce_max(n_clf, axis=3)
            n_clf_mask = tf.greater(n_clf_val, 0.5) # = where object was identified
            n_iou_mask = tf.greater(n_bnd[:,:,:,0], 0.2) # = where bbox was identified 

            n_bbox_indices = tf.where(tf.logical_and(n_clf_mask, n_iou_mask)) # "valid" bboxes

            #for (i, bbox_indices) in enumerate(tf.split(n_bbox_indices, NUM_BOXES, 2)):
            #n_bbox_locs = n_bbox_indices

            accuracy = tf.reduce_mean(tf.cast(tf.gather_nd( tf.equal(g_clf_pred, n_clf_pred), g_clf_indices),  tf.float32)) # just classification accuracy
            n_box = decode_box(n_bnd)

            box_with_clf = tf.concat([n_box, tf.cast(tf.expand_dims(n_clf_pred, -1),tf.float32)], axis=3) # 5xNUM_BOXES+1
            pred = tf.gather_nd(box_with_clf, n_bbox_indices)

            #box_pred = tf.gather_nd(n_bbox_indices,n_bnd) 

            #prediction = tf.greater(n_clf, 0.5)
            #ground_truth_prediction = tf.greater(g_clf, 0.5)
            #correct_prediction = tf.cast(tf.equal(prediction, ground_truth_prediction), tf.float32)
            #accuracy = tf.reduce_mean(correct_prediction)

    return pred, accuracy #bbox, accuracy 

def add_evaluation_steps(n_clfs, n_bnds, ground_truth_tensors):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  predictions = []
  accuracies = []
  i = 0
  n = len(n_clfs)
  for i in range(n): # iterate over different conv-layer classifiers
      with tf.name_scope('eval_%d' % i):
          gt = ground_truth_tensors[i]
          prediction, accuracy = add_evaluation_step(n_clfs[i], n_bnds[i], gt)
          #prediction, accuracy = add_evaluation_step(ft, gt)
          #predictions.append(prediction)
          accuracies.append(accuracy)
          predictions.append(prediction)
      i += 1
  accuracy = tf.reduce_mean(accuracies)
  
  tf.summary.scalar('accuracy', accuracy)
  pred = tf.concat(predictions, axis=0, name=FLAGS.final_tensor_name)
  print('ps', pred.shape)
  return pred, tf.reduce_mean(accuracy)


def create_feed_dict(ground_truth_inputs, ground_truths, bottleneck_inputs, bottlenecks):
    feed_dict = {}
    for gt, vgt in zip(ground_truth_inputs, ground_truths):
        feed_dict[gt] = vgt
    for bt, vbt in zip(bottleneck_inputs, bottlenecks):
        feed_dict[bt] = vbt
    return feed_dict

stop_request = False
def sigint_handler(signal, frame):
    global stop_request
    stop_request = True

def create_graph(graph_path):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def main(_):
  global stop_request
  signal.signal(signal.SIGINT, sigint_handler)


  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  graph, bottleneck_tensors, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

  #bottleneck_tensors = list

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.testing_percentage, FLAGS.validation_percentage)

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(graph=graph, config=config) as sess:

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop,
           FLAGS.random_scale, FLAGS.random_brightness)

    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_inputs, ground_truth_inputs, n_clfs, n_bnds) = add_final_training_ops(bottleneck_tensors)

    # Create the operations we need to evaluate the accuracy of our new layer.
    predictions, evaluation_step = add_evaluation_steps(n_clfs, n_bnds, ground_truth_inputs)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/validation')

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    print('START TRAINING !!')

    # Run the training for as many cycles as requested on the command line.
    def train(i):
        # Get a batch of input bottleneck values, either calculated fresh every
        # time with distortions applied, or from the cache stored on disk.
        if do_distort_images:
            (train_bottlenecks, train_ground_truths) = get_random_distorted_bottlenecks(
                 sess, image_lists, FLAGS.train_batch_size, 'training',
                 FLAGS.image_dir, distorted_jpeg_data_tensor,
                 distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            (train_bottlenecks, train_ground_truths, _) = get_random_cached_bottlenecks(
                 sess, image_lists, FLAGS.train_batch_size, 'training',
                 jpeg_data_tensor)
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.

        feed_dict = {}
        for gt, tgt in zip(ground_truth_inputs, train_ground_truths):
            feed_dict[gt] = tgt
        for bt, tbt in zip(bottleneck_inputs, train_bottlenecks):
            feed_dict[bt] = tbt

        train_summary, _ = sess.run(
            [merged, train_step],
            feed_dict=feed_dict) # if this doesn't work, manually construct feed_dict

        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict=feed_dict)
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                       cross_entropy_value))
            valid_bottlenecks, valid_ground_truths, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    jpeg_data_tensor))

            valid_feed_dict =  create_feed_dict(
                    ground_truth_inputs, valid_ground_truths,
                    bottleneck_inputs, valid_bottlenecks)

            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict=valid_feed_dict)
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (datetime.now(), i, validation_accuracy * 100,
                   len(valid_bottlenecks)))

    n = FLAGS.how_many_training_steps
    now = 0

    saver = tf.train.Saver()

    if FLAGS.load:
        if len(FLAGS.checkpoint_path) > 0:
            print("Loading from : %s" % FLAGS.checkpoint_path)
            saver.restore(sess, FLAGS.checkpoint_path)
        else:
            create_graph(FLAGS.input_graph)


    while True:
        for i in range(now, now + n):
            if stop_request:
                stop_request = False
                break
            train(i)
            if i>0 and (i % 1000) == 0:
                saver.save(sess, FLAGS.checkpoint_path)
                print("Model saved in : %s" % FLAGS.checkpoint_path)
        now += n
        s = raw_input('Enter Number of Episodes to Continue : \n')
        n = 0

        try:
            n = int(s)
        except Exception as e:
            break

    print('completed training !')
    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.

    test_bottlenecks, test_ground_truths, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', jpeg_data_tensor))

    print('create feed dict')
    test_feed_dict =  create_feed_dict(
            ground_truth_inputs, test_ground_truths,
            bottleneck_inputs, test_bottlenecks)

    print('Run Tests')
    test_accuracy, test_predictions = sess.run(
        [evaluation_step, predictions], feed_dict = test_feed_dict)


    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    #print(test_predictions)

    #if FLAGS.print_misclassified_test_images:
    #    print('=== MISCLASSIFIED TEST IMAGES ===')
    #    for i, test_filename in enumerate(test_filenames):
    #      if predictions[i] != test_ground_truth[i].argmax():
    #        print('%70s  %s' % (test_filename,
    #                          list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.

    print('Start Saving ... ')
    #tf.train.write_graph(graph.as_graph_def(), '', FLAGS.output_graph)  # graph def
    #saver.save(sess, FLAGS.checkpoint_path) # weights
    #freeze_graph(FLAGS.output_graph, '', False, FLAGS.checkpoint_path, output_node_names, 'ssd_save/restore', 'ssd_save/Const:0', FLAGS.output_graph, True,'','')

    #for n in output_nodes:
    #    print(n.name)

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=0,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=0,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
          '--checkpoint_path',
          type=str,
          default='/tmp/model.ckpt',
          help="Path To Save Model"
          )

  parser.add_argument(
          '--load',
          type=str2bool,
          nargs='?',
          const=True,
          default='n',
          help='Load Model From CheckPoint'
          )
  parser.add_argument(
          '--input_graph',
          type=str,
          default='',
          help="Input Graph (*.pb) to load from"
          )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
