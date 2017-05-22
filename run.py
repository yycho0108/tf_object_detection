"""
Preparing model:
 - Install bazel ( check tensorflow's github for more info )
    Ubuntu 14.04:
        - Requirements:
            sudo add-apt-repository ppa:webupd8team/java
            sudo apt-get update
            sudo apt-get install oracle-java8-installer
        - Download bazel, ( https://github.com/bazelbuild/bazel/releases )
          tested on: https://github.com/bazelbuild/bazel/releases/download/0.2.0/bazel-0.2.0-jdk7-installer-linux-x86_64.sh
        - chmod +x PATH_TO_INSTALL.SH
        - ./PATH_TO_INSTALL.SH --user
        - Place bazel onto path ( exact path to store shown in the output)
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph /path/output_graph.pb --output_labels /path/output_labels.txt --bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, change and run this code.
"""

import numpy as np
import tensorflow as tf
import cv2
import os

imagePath = 'backups/data/person/'
modelFullPath = 'backups/graph.pb'

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer


    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        for fdir in os.listdir(imagePath):
            f = os.path.join(imagePath, fdir)
            if os.path.isfile(f):
              image_data = tf.gfile.FastGFile(f, 'rb').read()
              predictions = sess.run(softmax_tensor,
                                     {'DecodeJpeg/contents:0': image_data})

              frame = cv2.imread(f)
              h,w,_ = frame.shape

              label = (predictions[:,1] > predictions[:,0]).reshape((8,8)).astype(np.uint8)*255
              label_frame = cv2.resize(label, (w,h), cv2.INTER_NEAREST)
              label_frame = cv2.cvtColor(label_frame, cv2.COLOR_GRAY2BGR)
              overlay = cv2.addWeighted(label_frame, 0.5, frame, 0.5, 0.0)

              cv2.imshow('frame', frame)
              cv2.imshow('label', label_frame)
              cv2.imshow('overlay', overlay)

              if cv2.waitKey(0) == 27:
                  return

if __name__ == '__main__':
    run_inference_on_image()
