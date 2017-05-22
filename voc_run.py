import numpy as np
import tensorflow as tf
import cv2
import os
from utils import non_max_suppression
WHITE = np.asarray([255,255,255], dtype=np.uint8)
colors = [WHITE]
for i in range(20):
    color = cv2.cvtColor(np.asarray([[[i, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
    colors.append(color)

imagePath = '/home/yoonyoungcho/Downloads/VOCdevkit/VOC2012/JPEGImages/'
modelFullPath = 'workspace/graph.pb'

def putText(frame, loc, txt):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(txt, font, 0.5, 0)[0]
    pt = (int(loc[0] - ts[0]/2.0), int(loc[1] - ts[1]/2.0))
    cv2.putText(frame, txt, pt, font, 0.5, (255,0,0))


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def rand_gen(high):
  while True:
    yield np.random.randint(high)

def report_graph(graph):
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

def run_inference_on_image():
    categories =[
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor'] 
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        report_graph(sess.graph)
        final_tensor = sess.graph.get_tensor_by_name('final_result:0')
        bnd_tensor = sess.graph.get_tensor_by_name('final_training_ops/n_split:1')
        cnt = 0
        fs = os.listdir(imagePath)
        for f_idx in rand_gen(len(fs)):
          fdir = fs[f_idx]
          f = os.path.join(imagePath, fdir)
          if os.path.isfile(f):
            image_data = tf.gfile.FastGFile(f, 'rb').read()

            predictions, bboxes = sess.run([final_tensor,bnd_tensor],
                                   {'DecodeJpeg/contents:0': image_data})

            #predictions = np.squeeze(predictions)
            predictions = predictions.reshape((8, 8, -1))
            predictions[predictions < 0.6] = 0.0

            frame = cv2.imread(f)
            h,w,_ = frame.shape

            h_box = h / 8.0
            w_box = w / 8.0

            # label = (8,8,21)
            #print predictions[0,0,:]
            label = np.argmax(predictions, axis=2)
            #print label
            label_rsz = cv2.resize(label, (w,h), interpolation=cv2.INTER_NEAREST)

            label_frame = np.zeros((h,w,3), dtype=np.uint8)
            for idx in range(21):
              indices = np.nonzero(label_rsz == idx)
              if len(indices[0]) > 0:
                print categories[idx]
              label_frame[label_rsz == idx] = colors[idx]

            rects = {}
            idxs = []

            for i in xrange(8):
                for j in xrange(8):
                    if label[i,j] == 0: # background
                        continue
                    idx = label[i,j]
                    x = (j + 0.5) * w_box
                    y = (i + 0.5) * h_box
                    dx,dy,dw,dh = bboxes[0,i,j,:]
                    dx *= w_box
                    dy *= h_box
                    dw *= w_box
                    dh *= h_box

                    cx = x + dx
                    cy = y + dy

                    x1 = int(cx - dw/2)
                    x2 = int(cx + dw/2)
                    y1 = int(cy - dh/2)
                    y2 = int(cy + dh/2)
                    idx = label[i,j]
                    if not rects.has_key(idx):
                        rects[idx] = [[x1,y1,x2,y2]]
                    else:
                        rects[idx].append([x1,y1,x2,y2])
            for idx in rects.keys():
                r = np.asarray(rects[idx])
                pick = non_max_suppression(r, 0.2)
                rects[idx] = r[pick]

            for idx in rects.keys():
                for rect in rects[idx]:
                    x1,y1,x2,y2 = rect
                    color = [int(val) for val in colors[idx]]
                    cv2.rectangle(frame,(x1,y1), (x2,y2), color = color, thickness = 2)
                    putText(frame, ((x1+x2)/2, (y1+y2)/2), categories[idx])

            overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)

            cv2.imshow('frame', frame)
            cv2.imshow('label', label_frame)
            cv2.imshow('overlay', overlay)

            k = cv2.waitKey(0)
            if k == 27:
                return
            elif k == ord('s'):
                cv2.imwrite(('image_%s.png' % cnt), overlay)
                cnt += 1

if __name__ == '__main__':
    run_inference_on_image()
