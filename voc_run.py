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

def rand_gen(high):
  while True:
    yield np.random.randint(high)

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
        final_tensor = sess.graph.get_tensor_by_name('final_result:0')
        cnt = 0
        fs = os.listdir(imagePath)
        for f_idx in rand_gen(len(fs)):
          fdir = fs[f_idx]
          f = os.path.join(imagePath, fdir)
          if os.path.isfile(f):
            image_data = tf.gfile.FastGFile(f, 'rb').read()
            predictions = sess.run(final_tensor,
                                   {'DecodeJpeg/contents:0': image_data})

            predictions = np.squeeze(predictions)
            predictions[predictions < 0.2] = 0.0

            frame = cv2.imread(f)
            h,w,_ = frame.shape

            # label = (x,8,8,21)
            #print predictions[0,0,:]
            label = np.argmax(predictions, axis=2)
            #print label
            label = cv2.resize(label, (w,h), interpolation=cv2.INTER_NEAREST)

            label_frame = np.zeros((h,w,3), dtype=np.uint8)
            for idx in range(21):
              indices = np.nonzero(label == idx)
              if len(indices[0]) > 0:
                print categories[idx]
              label_frame[label == idx] = colors[idx]

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
