from voc_utils import VOCLoader
import cv2

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import os
import numpy as np

import cv2

## VIZ PARAMS
WHITE = np.asarray([255,255,255], dtype=np.uint8)
colors = [WHITE]
for i in range(20):
    color = cv2.cvtColor(np.asarray([[[i, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
    colors.append(color)

BOTTLENECK_TENSOR_NAME = 'mixed_10/join:0' ## (1,8,8,2048)
BOTTLENECK_TENSOR_SIZE = 2048

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 480
MODEL_INPUT_DEPTH = 3

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

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
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))

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

def overlap(r_a, r_b):
    xa1,ya1,xa2,ya2 = r_a
    xb1,yb1,xb2,yb2 = r_b
    return max(0, min(xa2,xb2) - max(xa1,xb1)) * max(0, min(ya2,yb2) - max(ya1,yb1))

def create_label(ann, categories):
    width = int(ann.findChild('width').contents[0])
    height = int(ann.findChild('height').contents[0])

    num_classes = len(categories) + 1 # 0 = background

    w_box = width / 8.0
    h_box = height / 8.0

    objs = ann.findAll('object')

    label = np.zeros((8,8,num_classes), dtype=np.float32)

    #bbox_label = np.zeros((8,8,num_classes,4), dtype=np.float32)

    label[:, :, 0] = 1.0 # == background

    for obj in objs:
        category = obj.findChild('name').contents[0]

        # 0 = background
        idx = categories.index(category)
        box = obj.findChild('bndbox')

        xmin = int(box.findChild('xmin').contents[0])
        ymin = int(box.findChild('ymin').contents[0])
        xmax = int(box.findChild('xmax').contents[0])
        ymax = int(box.findChild('ymax').contents[0])

        for i in range(8):
            for j in range(8):
              box1 = (j * w_box, i * h_box, (j+1) * w_box, (i+1) * h_box)
              box2 = (xmin, ymin, xmax, ymax)
              o = overlap(box1, box2) / (w_box * h_box)
              if o > 0:
                label[i,j,idx] += o
                label[i,j,0] = 0

        #xmin = int(np.floor(xmin / w_box))
        #ymin = int(np.floor(ymin / h_box))
        #xmax = int(np.ceil(xmax / w_box))
        #ymax = int(np.ceil(ymax / h_box))

        #label[ymin:ymax, xmin:xmax, idx] = 1.0
        #label[ymin:ymax, xmin:xmax, 0] = 0.0

    return label

def process():
    btl_dir = './workspace/bottlenecks'

    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

    loader = VOCLoader('/home/jamiecho/Downloads/VOCdevkit/VOC2012/')

    categories = loader.list_image_sets()
    num_classes = len(categories)+1

    with tf.Session(graph = graph) as sess:
        cnt = 0
        for ann in loader.annotations():
            cnt += 1
            print cnt
            # bottleneck
            img = loader.img_from_annotation(ann)

            # paths
            common = os.path.splitext(os.path.basename(img))[0]

            bv_path = os.path.join(btl_dir, common + '_btl.npy')
            bl_path = os.path.join(btl_dir, common + '_lbl.npy')

            if os.path.exists(bl_path):
                continue

            # run label
            label = create_label(ann, categories)
            np.save(bl_path, label, allow_pickle=True)

            if os.path.exists(bv_path):
                continue

            # run bottleneck
            image_data = gfile.FastGFile(img, 'rb').read()
            bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
            # save
            np.save(bv_path, bottleneck_values, allow_pickle=True)

            #VIZ
            #frame = cv2.imread(img)
            #label_frame = np.zeros((8,8,3), dtype=np.uint8)
            #for idx in range(num_classes):
            #    label_frame[label[:,:,idx] == 1] = colors[idx]
            #h,w = frame.shape[:-1]
            #label_frame = cv2.resize(label_frame, (w,h), cv2.INTER_LINEAR)
            #cv2.imshow('label', label_frame)
            #cv2.imshow('frame', frame)
            #overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)
            #cv2.imshow('overlay', overlay)
            #if cv2.waitKey(0) == 27:
            #    return

def visualize():
  loader = VOCLoader('/home/jamiecho/Downloads/VOCdevkit/VOC2012/')
  categories = ['background'] + loader.list_image_sets()
  num_classes = len(categories)

  for ann in loader.annotations():
    img = loader.img_from_annotation(ann)
    print img

    label = create_label(ann, categories)
    label_frame = np.zeros((8,8,3), dtype=np.uint8)

    frame = cv2.imread(img)
    h,w = frame.shape[:-1]

    for idx in range(num_classes):
      indices = (label[:,:,idx] != 0)
      label_frame[indices,:] = colors[idx]
      if (len(np.nonzero(indices)[0]) > 0):
        print categories[idx]

    label_frame = cv2.resize(label_frame, (w,h), cv2.INTER_LINEAR)
    cv2.imshow('label', label_frame)
    cv2.imshow('frame', frame)
    overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)
    cv2.imshow('overlay', overlay)
    if cv2.waitKey(0) == 27:
        return

if __name__ == "__main__":
    #process()
    visualize()
