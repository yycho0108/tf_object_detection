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

from ast import literal_eval as make_tuple

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
#BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

BOTTLENECK_TENSOR_NAME = 'mixed_10/join:0' ## (1,8,8,2048)
BOTTLENECK_TENSOR_SIZE = 2048

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 480
MODEL_INPUT_DEPTH = 3

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def get_immediate_subdirectories(a_dir):
  res = []
  for sub in os.listdir(a_dir):
    d = os.path.join(a_dir, sub)
    if os.path.isdir(d):
      res.append(d)
  return res

def get_image(d, entry):
  im_path = entry.split(': ')[0].replace('"','')
  fn, ext= os.path.splitext(im_path)
  return os.path.join(d, fn)

def create_image_lists(image_path, testing_percentage, validation_percentage):

  result = {}

  training_images = []
  testing_images = []
  validation_images = []

  for sub in get_immediate_subdirectories(image_path):
    folder_name = os.path.basename(sub)
    bottleneck_dirname = os.path.join(FLAGS.bottleneck_dir, folder_name)
    annot_path = os.path.join(sub,'annotation.idl')

    with open(annot_path) as f:
      entries = f.read().split(';\n')[:-1]#remove last
      image_files = map(lambda entry : get_image(bottleneck_dirname, entry), entries)
      for image_file in image_files:
        base_name = os.path.basename(image_file)
        hash_name = re.sub(r'_nohash_.*$', '', image_file)
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

    result[sub] = {
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.npy'


def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))

  for op in graph.get_operations():
     print('===')
     print(op.name)
     print('Input:')
     for i in op.inputs:
         print('\t %s' % i.name, i.get_shape())
     print('Output:')
     for o in op.outputs:
         print('\t %s' % o.name, o.get_shape())
     print('===')

  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract():
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def create_bottleneck_file(sess, d, bottleneck_root, jpeg_data_tensor, bottleneck_tensor, box_ext='_box.npy', lab_ext='_lab.npy'):
  idl = os.path.join(d,'annotation.idl')
  with open(idl) as f:
    entries = f.read().split(';\n')
    cnt = 0
    for entry in entries:

      #if cnt % 10 == 0:
      #  print(cnt)
      #cnt += 1

      if entry:

        im_path, bbox = entry.split(': ')
        im_path = im_path.replace('"', '')
        im_path = os.path.join(d,im_path)

        fn, ext = os.path.splitext(im_path)
        fn = os.path.basename(fn)
        bv_path = os.path.join(bottleneck_root, fn + box_ext)
        bl_path = os.path.join(bottleneck_root, fn + lab_ext)

        if os.path.exists(bv_path) and os.path.exists(bl_path):
          continue

        image_data = gfile.FastGFile(im_path, 'rb').read()

        try:
          bottleneck_values = run_bottleneck_on_image(
              sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        except:
          raise RuntimeError('Error during processing file %s' % im_path)

        h_box = 480/8.
        w_box = 640/8.
        bbox = make_tuple(bbox)
        bottleneck_labels = np.full((8,8,2), [1,0], dtype=np.float32) # class 0 : background
        for box in bbox:
          x1,y1 = box[0], box[1]
          x2,y2 = box[2], box[3]
          bottleneck_labels[int(np.floor(y1/h_box)):int(np.ceil(y2/h_box)),int(np.floor(x1/w_box)):int(np.ceil(x2/w_box)),:] = [0,1] # class 1 : person

        np.save(bv_path, bottleneck_values, allow_pickle=True)
        np.save(bl_path, bottleneck_labels, allow_pickle=True)

def get_or_create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor):
  global flag
  box_ext='_box.npy'
  lab_ext='_lab.npy'

  ## CREATE IF DOESN'T EXIST
  for d in get_immediate_subdirectories('./data'):
    folder_name = os.path.basename(os.path.normpath(d))
    if not os.path.exists(os.path.join(FLAGS.bottleneck_dir, folder_name)):
      create_bottleneck_file(sess, d, FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor, box_ext, lab_ext)

  fn, ext = os.path.splitext(image_name)
  bv_path = os.path.join(FLAGS.bottleneck_dir, fn + box_ext)
  bl_path = os.path.join(FLAGS.bottleneck_dir, fn + lab_ext)

  val = np.load(bv_path, allow_pickle=True)
  lab = np.load(bl_path, allow_pickle=True)
  return val, lab

def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, image_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor)
        #get_or_create_bottleneck(sess, image_lists, label_name, index,
        #                         image_dir, category, bottleneck_dir,
        #                         jpeg_data_tensor, bottleneck_tensor)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
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
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      l = image_lists[label_name][category]
      image_name = l[image_index % len(l)]
      bottleneck, ground_truth = get_or_create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor)
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):

        l = image_lists[label_name][category]
        image_name = l[image_index % len(l)]
        bottleneck, ground_truth = get_or_create_bottleneck(sess, image_name, jpeg_data_tensor, bottleneck_tensor)
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

#def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
#                                  bottleneck_dir, image_dir, jpeg_data_tensor,
#                                  bottleneck_tensor):


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

def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):

  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  #bottleneck_tensor = tf.squeeze(bottleneck_tensor)
  # now (8x8x2048)

  with tf.name_scope('input'):
    shape = [None] + bottleneck_tensor.shape.dims
    print('s', shape)
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None,8,8,2048],
        name='BottleneckInputPlaceholder')

    # for classification
    ground_truth_input = tf.placeholder(tf.float32,
        [None, 8, 8, 2],
        name='GroundTruthInput')

    # for bounding boxes!
    #ground_truth_bbox_input = tf.placeholder(tf.float32,
    #    [None, 8, 8, 4],
    #    name='GroundTruthBBoxInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      # (batchx8x8x2048) --> (batchx8x8xclass_count)
      initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 2],
                                          stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([8, 8, 2]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      #logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      logits = tf.tensordot(bottleneck_input, layer_weights, axes=[[3],[0]]) + layer_biases
      # --> logits for presence

      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
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
      prediction = tf.greater(result_tensor, 0.5)
      ground_truth_prediction = tf.greater(ground_truth_tensor, 0.5)
      correct_prediction = tf.cast(tf.equal(prediction, ground_truth_prediction), tf.float32)
      evaluation_step = tf.reduce_mean(correct_prediction)
  tf.summary.scalar('accuracy', evaluation_step)
  return prediction, evaluation_step


def main(_):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  #if class_count == 1:
  #  print('Only one valid folder of images found at ' + FLAGS.image_dir +
  #        ' - multiple classes are needed for classification.')
  #  return -1

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)

  with tf.Session(graph=graph) as sess:

    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop,
           FLAGS.random_scale, FLAGS.random_brightness)
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      pass
      #cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
      #                  FLAGS.bottleneck_dir, jpeg_data_tensor,
      #                  bottleneck_tensor)

    # Add the new layer that we'll be training.
    #(train_step, cross_entropy, bottleneck_input, ground_truth_input,
    # final_tensor) = add_final_training_ops(len(image_lists.keys()),
    #                                        FLAGS.final_tensor_name,
    #                                        bottleneck_tensor)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # Create the operations we need to evaluate the accuracy of our new layer.
    prediction, evaluation_step = add_evaluation_step(
        final_tensor, ground_truth_input)

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
    for i in range(FLAGS.how_many_training_steps):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor,
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             bottleneck_tensor)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.

      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                        train_accuracy * 100))
        print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                   cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                bottleneck_tensor))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
              (datetime.now(), i, validation_accuracy * 100,
               len(validation_bottlenecks)))

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    #if FLAGS.print_misclassified_test_images:
    #  print('=== MISCLASSIFIED TEST IMAGES ===')
    #  for i, test_filename in enumerate(test_filenames):
    #    if predictions[i] != test_ground_truth[i].argmax():
    #      print('%70s  %s' % (test_filename,
    #                          list(image_lists.keys())[predictions[i]]))

    # Write out the trained graph and labels with the weights stored as
    # constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')


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
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
