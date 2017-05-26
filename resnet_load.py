import tensorflow as tf
slim = tf.contrib.slim

from tensorflow.contrib.slim.nets import resnet_utils, resnet_v2
from inception_preprocessing import preprocess_for_eval as proc

import cv2
import numpy as np

from imagenet_labels import lab as label_names
def cap():
    cam = cv2.VideoCapture(0)
    for i in range(10):
        cam.read()
    _,bgr = cam.read()
    return bgr[...,::-1]

OUTPUT_TENSOR_NAMES = [('resnet_v2_101/block%d/unit_1/bottleneck_v2/preact/batchnorm/mul_1:0' % i)
        for i in range(1,5)] +
def create_graph():


def main():
    input = tf.placeholder(tf.float32, [None,None,3])
    rsz = tf.expand_dims(proc(input, 299, 299), 0)

    with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training = False)):
        net, end_points = resnet_v2.resnet_v2_101(
                rsz, 1001, global_pool=False)
        print net.shape
        #print end_points

    # 0 = background
    # 1 ~ 1000 = imagenet labels

    labels = tf.squeeze(tf.argmax(net, axis=3))
    values = tf.squeeze(tf.reduce_max(net, axis=3))


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'workspace/resnet/resnet_v2_101.ckpt')

        writer = tf.summary.FileWriter('/tmp/resnet', sess.graph)

        cam = cv2.VideoCapture(0)
        for i in range(30):
            cam.read()

        s = tf.expand_dims(tf.cast(labels, tf.float32)/1001., -1)
        s = tf.expand_dims(s, 0)
        im = tf.summary.image('labels', s)

        cnt = 0
        while True:
            cnt += 1
            #bgr = cv2.imread('guac.jpg')
            #print bgr.shape
            _,bgr = cam.read()
            rgb = bgr[...,::-1]/255.
            im, res_lab, res_val = sess.run([im, labels,values], feed_dict={input : rgb})
            writer.add_summary(im, cnt)

            h,w,_ = rgb.shape
            print res_lab
            idx = np.unravel_index(np.argmax(res_val), res_val.shape)
            label_frame = cv2.resize(res_lab/1001., (w,h), cv2.INTER_LINEAR)
            clf = np.argmax(np.bincount(res_lab.reshape(-1)))

            mask = (res_lab == clf)
            print mask

            print label_names[clf-1]
            #print label_names[res_lab[idx]]

            cv2.imshow('frame', bgr)
            #cv2.imshow('masked', bgr * (mask[:,:,None].resize(w,h).astype(bgr.dtype)) )
            #cv2.imshow('label', label_frame)
            if cv2.waitKey(0) == 27:
                break
if __name__ == "__main__":
    main()
