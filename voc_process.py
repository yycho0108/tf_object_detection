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

from config import *


## VIZ PARAMS
WHITE = np.asarray([255,255,255], dtype=np.uint8)
colors = [WHITE]
for i in range(20):
    color = cv2.cvtColor(np.asarray([[[i, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
    colors.append([int(c) for c in color])

# mixed_2/join : 35
# mixed_7/join : 17
# mixed_10/join : 8

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

            return_elements = list(OUTPUT_TENSOR_NAMES)
            return_elements += [JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]
            results = tf.import_graph_def(graph_def, name='', return_elements = return_elements)

            output_tensors = results[:-2]

            ## ADD A POOL ...
            for i, k in enumerate(APPEND_POOL):
                name = ('aux_pool_%d' % i)
                p = tf.nn.max_pool(output_tensors[-1],ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
                output_tensors.append(p)

            #for t in output_tensors:
            #    print(t.name,t.shape)

            jpeg_data_tensor = results[-2]
            resized_input_tensor = results[-1]
                    
    return graph, output_tensors, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, output_tensors):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  output_values = sess.run(output_tensors, {image_data_tensor : image_data})
  output_values = [np.squeeze(v) for v in output_values]
  return output_values
  #bottleneck_values = sess.run(
  #    bottleneck_tensor,
  #    {image_data_tensor: image_data})
  #bottleneck_values = np.squeeze(bottleneck_values)
  #return bottleneck_values

flag1 = False
flag2 = False

def overlap(r_a, r_b):
    xa1,ya1,xa2,ya2 = r_a
    xb1,yb1,xb2,yb2 = r_b
    return max(0, min(xa2,xb2) - max(xa1,xb1)) * max(0, min(ya2,yb2) - max(ya1,yb1))

def create_ssd_label(ann, categories, output_shapes):
    width = float(ann.findChild('width').contents[0])
    height = float(ann.findChild('height').contents[0])

    num_classes = len(categories)
    num_boxes = len(BBOX_RATIOS)
    output_dims = num_classes + 5 * num_boxes 

    objs = ann.findAll('object')

    labels = []
    ious = []

    for s in output_shapes:
        n, m = s
        label_s = np.zeros((n,m, output_dims), dtype=np.float32)
        iou_s = np.zeros((n,m, num_boxes), dtype=np.float32)

        for obj in objs:
            category = obj.findChild('name').contents[0]
            idx = categories.index(category)
            box = obj.findChild('bndbox')

            xmin = float(box.findChild('xmin').contents[0])
            ymin = float(box.findChild('ymin').contents[0])
            xmax = float(box.findChild('xmax').contents[0])
            ymax = float(box.findChild('ymax').contents[0])

            ref_box = (xmin, ymin, xmax, ymax)
            ref_w = (xmax - xmin)
            ref_h = (ymax - ymin)

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            w_box = width / m
            h_box = height / n

            for i in range(n):
                for j in range(m):
                    cell_box = (j * w_box, i * h_box, (j+1) * w_box, (i+1) * h_box)
                    o = overlap(ref_box, cell_box) / (w_box * h_box)
                    label_s[i,j,idx] += o

                    x = (j+0.5) * w_box
                    y = (i+0.5) * h_box

                    #default boxes
                    for box_idx, ratio in enumerate(BBOX_RATIOS):
                        w_r, h_r = ratio
                        w = w_box * w_r
                        h = h_box * h_r

                        box_i = (x-w/2,y-h/2,x+w/2,y+h/2)

                        o_i = overlap(ref_box, box_i)
                        u_i = w*h + ref_w * ref_h - o_i
                        iou_i = o_i / u_i  # == IOU

                        base_idx = num_classes + 5 * box_idx
                        if(i == 0 and j == 0):
                            dx = (cx - x) / w_box
                            dy = (cy - y) / h_box
                            dw = (xmax - xmin) / w 
                            dh = (ymax - ymin) / h 
                        if iou_i > iou_s[i,j,box_idx]:
                            # update
                            dx = (cx - x) / w_box
                            dy = (cy - y) / h_box
                            dw = (xmax - xmin) / w 
                            dh = (ymax - ymin) / h 

                            iou_s[i,j,box_idx] = iou_i # remember max value
                            label_s[i,j,base_idx:base_idx+5] = iou_i, dx, dy, dw, dh
                        #print 'iou', iou_i
                        #print 'dx', dx
                        #print 'dy', dy
                        #print 'dw', dw
                        #print 'dh', dh



        labels.append(label_s)
    return labels

def process():
    btl_dir = './workspace/bottlenecks'
    if not os.path.isdir(btl_dir):
        os.makedirs(btl_dir)
        

    graph, output_tensors, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

    loader = VOCLoader('/home/yoonyoungcho/Downloads/VOCdevkit/VOC2012/')

    categories = loader.list_image_sets()
    num_classes = len(categories)

    with tf.Session(graph = graph) as sess:
        for op in graph.get_operations():
            print('=====')
            print(op.name)
            print('\t Input :')
            for i in op.inputs:
                print('\t \t %s %s' % (i.name, str(i.shape)))
            print('\t Output:')
            for o in op.outputs:
                print('\t \t %s %s' % (o.name, str(o.shape)))

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
            output_shapes = [t.shape.as_list()[1:3] for t in output_tensors]
            labels = create_ssd_label(ann, categories, output_shapes)
            np.save(bl_path, labels, allow_pickle=True)

            if os.path.exists(bv_path):
                continue

            # run bottleneck
            image_data = gfile.FastGFile(img, 'rb').read()
            output_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, output_tensors)
            # save
            np.save(bv_path, output_values, allow_pickle=True)

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
  loader = VOCLoader('/home/yoonyoungcho/Downloads/VOCdevkit/VOC2012/')
  categories = loader.list_image_sets()
  num_classes = len(categories)

  for ann in loader.annotations():
    img = loader.img_from_annotation(ann)
    print img
    output_shapes = [[35,35], [17,17], [8,8], [4,4], [2,2]]

    labels = create_ssd_label(ann, categories, output_shapes)
    label_frames = []

    frame = cv2.imread(img)
    h,w = frame.shape[:-1]

    label_frames = []
    for label in labels: # label per-bbox-aspects
        rects = []
        reg_rects = []

        n,m = label.shape[:2]
        label_frame = np.zeros((n,m,3), dtype=np.uint8)

        w_box = float(w) / m
        h_box = float(h) / n

        for i in range(n):
            for j in range(m):
                idx = np.argmax(label[i, j, :num_classes])
                val = label[i,j,idx]
                #if val < 0.5: # classification confidence
                #    continue
                label_frame[i,j,:] = colors[idx]
                ncc = num_classes+1
                bboxes = label[i,j,num_classes:].reshape(-1,5)
                for box_idx, bbox in enumerate(bboxes):
                    iou_i, dx, dy, dw, dh = bbox
                    if iou_i < 0.2: # localization confidence
                        continue
                    b_x = w_box * (j+0.5)
                    b_y = h_box * (i+0.5)

                    w_r,h_r = BBOX_RATIOS[box_idx]
                    b_w = w_box * w_r
                    b_h = h_box * h_r

                    p1 = (int(b_x - b_w/2), int(b_y - b_h/2))
                    p2 = (int(b_x + b_w/2), int(b_y + b_h/2))
                    rects.append((p1, p2,colors[idx]))
                    b_x_r = b_x + dx*w_box
                    b_y_r = b_y + dy*h_box
                    b_w_r = b_w * dw
                    b_h_r = b_h * dh 

                    p1_r = (int(b_x_r - b_w_r/2), int(b_y_r - b_h_r/2))
                    p2_r = (int(b_x_r + b_w_r/2), int(b_y_r + b_h_r/2))
                    reg_rects.append((p1_r, p2_r, colors[idx]))

                    
        label_frame = cv2.resize(label_frame, (int(np.ceil(w)),int(np.ceil(h))), cv2.INTER_LINEAR)
        for i in range(len(rects)):
            #print reg_rects[i]
            #print rects[i]
            p1, p2, col = rects[i]
            cv2.rectangle(frame, p1, p2, (255,0,0), 2)
            p1, p2, col = reg_rects[i]
            cv2.rectangle(frame, p1, p2, [int(c) for c in col], 2)
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(0) == 27:
            #    return
        label_frames.append(label_frame)

    cv2.imshow('frame', frame)
    for i, label_frame in enumerate(label_frames):
        cv2.imshow(('label_%d' % i), label_frame)
    if cv2.waitKey(0) == 27:
        return


    #for idx in range(num_classes):
    #  indices = (label[:,:,idx] != 0)
    #  label_frame[indices,:] = colors[idx]
    #  if (len(np.nonzero(indices)[0]) > 0):
    #    print categories[idx]

    #label_frame = cv2.resize(label_frame, (w,h), cv2.INTER_LINEAR)
    #cv2.imshow('label', label_frame)
    #cv2.imshow('frame', frame)
    #overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)
    #cv2.imshow('overlay', overlay)
    #if cv2.waitKey(0) == 27:
    #    return

if __name__ == "__main__":
    process()
    #visualize()
