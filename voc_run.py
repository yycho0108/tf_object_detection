import numpy as np
import tensorflow as tf
import cv2
import os

WHITE = np.asarray([255,255,255], dtype=np.uint8)
colors = [WHITE]
for i in range(20):
    color = cv2.cvtColor(np.asarray([[[i, 255, 255]]],dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0]
    colors.append(color)

imagePath = '/home/jamiecho/Downloads/VOCdevkit/VOC2012/JPEGImages/'
modelFullPath = 'workspace/graph.pb'

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image():
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        final_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for fdir in os.listdir(imagePath):
            f = os.path.join(imagePath, fdir)
            if os.path.isfile(f):
              image_data = tf.gfile.FastGFile(f, 'rb').read()
              predictions = sess.run(final_tensor,
                                     {'DecodeJpeg/contents:0': image_data})
              # squeeze?
              print 'ps', predictions.shape

              frame = cv2.imread(f)
              h,w,_ = frame.shape

              # label = (x,8,8,21)
              label = np.argmax(label, axis=2)

              label_frame = np.zeros((w,h,3), dtype=np.uint8)
              for idx in range(21):
                label_frame[label == idx] = color[idx]

              overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)

              cv2.imshow('frame', frame)
              cv2.imshow('label', label_frame)
              cv2.imshow('overlay', overlay)

              if cv2.waitKey(0) == 27:
                  return

if __name__ == '__main__':
    run_inference_on_image()
