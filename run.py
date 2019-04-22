from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from imutils import face_utils
from source import inception_resnet_v1
from source.ThinPlateSpline2 import ThinPlateSpline2 as TPS
from source.attack_util import perturb
import matplotlib

matplotlib.use('Agg')

DEFAULT_IMAGE_SIZE = 182
DEFAULT_EMBEDDING_SIZE = 128
DEFAULT_EPSILON = 0.005
DEFAULT_N_ITER = 20

def parseargs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str,
                        default=r'C:\Users\Sanjay Saha\StealthyFacePerturbation\pretrained\20180408-102900\model'
                                r'-20180408-102900.ckpt-90')
    parser.add_argument('--dlib_model', type=str, default=r'C:\Users\Sanjay Saha\StealthyFacePerturbation\pretrained'
                                                          r'\shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--model_def', type=str, default='source.inception_resnet_v1')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--embedding_size', type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument('--use_fixed_image_standardization', action='store_true')
    parser.add_argument('--prelogits_hist_max', type=float, default=10.0)
    parser.add_argument('--epsilon', type=float, default=0.005)
    parser.add_argument('--img', type=str, default=r'C:\Users\Sanjay Saha\StealthyFacePerturbation\input\37.png')
    parser.add_argument('--true_label', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='.\output')
    parser.add_argument('--fixed_points', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1311)
    return parser.parse_args(argv)


def image_warping(img, lndA, lndB):
    CROP_SIZE = 182
    input_images_expanded = tf.reshape(img, [1, CROP_SIZE, CROP_SIZE, 3, 1])
    t_img, T, det = TPS(input_images_expanded, lndA, lndB, [CROP_SIZE, CROP_SIZE, 3])
    t_img = tf.reshape(t_img, [1, CROP_SIZE, CROP_SIZE, 3])
    return t_img, T


def main(args):
    np.random.seed(seed=args.seed)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.dlib_model)

    network = inception_resnet_v1
    y = tf.placeholder(tf.int32)

    lnd_A = tf.placeholder(tf.float32, [None, 2])
    lnd_B = tf.placeholder(tf.float32, [None, 2])

    x = tf.placeholder(tf.float32, shape=[182, 182, 3])
    images = x
    images = tf.image.per_image_standardization(images)
    images = tf.reshape(images, [-1, 182, 182, 3])

    lnd_source = tf.expand_dims(lnd_A, axis=0)
    lnd_target = tf.expand_dims(lnd_B, axis=0)

    images_deformed, T = image_warping(images, lnd_target, lnd_source)
    images_deformed = tf.image.per_image_standardization(images_deformed[0])
    images_deformed = tf.expand_dims(images_deformed, axis=0)

    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Trained model: %s' % pretrained_model)
    else:
        exit("A pretrained model should be provided!")

    tf.set_random_seed(args.seed)

    # Build the inference graph
    prelogits, cam_conv, _ = network.inference(images_deformed, 1.,  phase_train=False, bottleneck_layer_size=512)
    logits = slim.fully_connected(prelogits, 10575, activation_fn=None,
                                  scope='Logits', reuse=False)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    softmax = tf.nn.softmax(logits, axis=1)
    grad = tf.gradients(loss, lnd_B)[0] * 1.

    # Create a saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # load checkpoint
        if tf.train.checkpoint_exists(pretrained_model):
            print('Restoring pretrained model: %s' % pretrained_model)
            saver.restore(sess, pretrained_model)
        else:
            print('There is no checkpoint to load!')

        # read input image
        img = cv2.imread(args.img)
        img = cv2.resize(img, (182, 182))

        # convert color image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # extract landmarks
        rect = detector(gray, 1)[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # generate edge points to force the transformation to keep the boundary
        step = args.fixed_points
        new_w = 182
        steps = np.array(list(range(0, new_w, new_w // step)))
        b = list()
        for s in steps:
            b.append([0, s])
            b.append([s, 0])
            b.append([182, s])
            b.append([s, 182])
        b = np.array(b)
        b = b.reshape([-1, 2])
        shape = np.concatenate((shape, b), axis=0)
        lnd = np.copy(shape)

        # convert to rgb
        img = img[..., ::-1]

        # remove mean and scale pixel values
        img = (img - 127.5) / 128.

        # scale landmarks to [-1, 1]
        dp = np.copy((lnd / 182.) * 2. - 1.)
        lnd = np.copy(dp)

        # initialize the landmark locations of the adversarial face image
        lnd_adv = np.copy(lnd)

        print('True label:', args.true_label)
        for i in range(DEFAULT_N_ITER):
            l, s, img_d, t, grad_ = sess.run([logits, softmax, images_deformed, T, grad],
                                             feed_dict={x: img, lnd_A: lnd, lnd_B: lnd_adv, y: [args.true_label]})
            print("step: %02d, Predicted class: %05d, Pr(predicted class): %.4f, Pr(true class): %.4f" % (
            i, np.argmax(l), s.max(), s[0, args.true_label]))

            epsilon = args.epsilon
            lnd_adv = perturb(lnd_adv, grad=grad_, epsilon=epsilon)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        img_d = img_d.reshape([182, 182, 3])

        def prepare_for_save(x):
            x = (x - x.min()) / (x.max() - x.min())
            x = (x * 255.).astype(np.uint8)
            x = x[..., ::-1]
            return x

        # save output images
        cv2.imwrite(os.path.join(args.output_dir, "original.png"), prepare_for_save(img))
        cv2.imwrite(os.path.join(args.output_dir, "perturbed.png"), prepare_for_save(img_d))

        print('Please checck the output directory for the flow and the transformed image')


if __name__ == '__main__':
    main(parseargs(sys.argv[1:]))
