########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import paddle.v2 as paddle
import numpy as np
import argparse
import time

DTYPE = tf.float32
parser = argparse.ArgumentParser("VGG16 benchmark.")
parser.add_argument(
    '--batch_size', type=int, default=32, help="Batch size for training.")
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--num_passes', type=int, default=10, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    '--data_format',
    type=str,
    default='NHWC',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
parser.add_argument(
    '--num_skip_batch',
    type=int,
    default=0,
    help='The first #num_skip_batch batches'
    'will be skipped for timing.')
parser.add_argument(
    '--iterations', type=int, default=0, help='Maximum iterations')
args = parser.parse_args()


class VGG16Model(object):
    def infer(self, imgs, weights=None, sess=None):
        self.probs = tf.nn.softmax(self.network(imgs))
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        #TODO(add output here)

    def conv_layer(self, name, images, kernel_shape):
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(
                tf.truncated_normal(
                    kernel_shape, dtype=tf.float32, stddev=1e-1),
                name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(
                tf.constant(
                    0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
                trainable=True,
                name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            return out

    def fc_layer(self, name, inputs, shape):
        with tf.name_scope(name) as scope:
            fc_w = tf.Variable(
                tf.truncated_normal(
                    shape, dtype=tf.float32, stddev=1e-1),
                name='weights')
            fc_b = tf.Variable(
                tf.constant(
                    0.0, shape=[shape[-1]], dtype=tf.float32),
                trainable=True,
                name='biases')
            fc_l = tf.nn.bias_add(tf.matmul(inputs, fc_w), fc_b)
            out = tf.nn.relu(fc_l)
            self.parameters += [fc_w, fc_b]
            return fc_l, out

    def network(self, images, mode="training"):
        self.parameters = []

        # conv1_1
        self.conv1_1 = self.conv_layer('conv1_1', images, [3, 3, 3, 64])
        # conv1_2
        self.conv1_2 = self.conv_layer('conv1_2', self.conv1_1, [3, 3, 64, 64])
        # pool1
        self.pool1 = tf.nn.max_pool(
            self.conv1_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1')
        # conv2_1
        self.conv2_1 = self.conv_layer('conv2_1', self.pool1, [3, 3, 64, 128])
        # conv2_2
        self.conv2_2 = self.conv_layer('conv2_2', self.conv2_1,
                                       [3, 3, 128, 128])
        # pool2
        self.pool2 = tf.nn.max_pool(
            self.conv2_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2')
        # conv3_1
        self.conv3_1 = self.conv_layer('conv3_1', self.pool2, [3, 3, 128, 256])
        # conv3_2
        self.conv3_2 = self.conv_layer('conv3_2', self.conv3_1,
                                       [3, 3, 256, 256])
        # conv3_3
        self.conv3_3 = self.conv_layer('conv3_3', self.conv3_2,
                                       [3, 3, 256, 256])
        # pool3
        self.pool3 = tf.nn.max_pool(
            self.conv3_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool3')
        # conv4_1
        self.conv4_1 = self.conv_layer('conv4_1', self.pool3, [3, 3, 256, 512])
        # conv4_2
        self.conv4_2 = self.conv_layer('conv4_2', self.conv4_1,
                                       [3, 3, 512, 512])
        # conv4_3
        self.conv4_3 = self.conv_layer('conv4_3', self.conv4_2,
                                       [3, 3, 512, 512])
        # pool4
        self.pool4 = tf.nn.max_pool(
            self.conv4_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4')
        # conv5_1
        self.conv5_1 = self.conv_layer('conv5_1', self.pool4, [3, 3, 512, 512])
        # conv5_2
        self.conv5_2 = self.conv_layer('conv5_2', self.conv5_1,
                                       [3, 3, 512, 512])
        # conv5_3
        self.conv5_3 = self.conv_layer('conv5_3', self.conv5_2,
                                       [3, 3, 512, 512])
        # pool5
        self.pool5 = tf.nn.max_pool(
            self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool4')
        # flatten
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        # fc1
        _, self.fc1 = self.fc_layer('fc1', pool5_flat, [shape, 512])
        # fc2
        _, self.fc2 = self.fc_layer('fc2', self.fc1, [512, 512])
        # fc3
        self.fc3l, _ = self.fc_layer('fc3', self.fc2, [512, 102])

        return self.fc3l
        #return tf.nn.softmax(self.fc3l)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))


def run_benchmark():
    class_dim = 102
    dshape = (None, 224, 224, 3)
    device = '/cpu:0' if args.device == 'CPU' else '/device:GPU:0'
    with tf.device(device):
        images = tf.placeholder(DTYPE, shape=dshape)
        labels = tf.placeholder(tf.int64, shape=(None, ))
        onehot_labels = tf.one_hot(labels, depth=class_dim)

        model = VGG16Model()
        logits = model.network(images)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        avg_loss = tf.reduce_mean(loss)

        correct = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        g_accuracy = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_op = optimizer.minimize(avg_loss)

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=5120),
        batch_size=args.batch_size)

    with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)
        iters, num_samples = 0, 0
        for pass_id in range(args.num_passes):
            if args.iterations == iters:
                break
            for batch_id, data in enumerate(train_reader()):
                if args.num_skip_batch == iters:
                    start_time = time.clock()
                train_images = np.array(
                    map(lambda x: np.transpose(x[0].reshape([3, 224, 224]),
                    axes=[1, 2, 0]), data)).astype("float32")
                train_labels = np.array(map(lambda x: x[1], data)).astype(
                    'int64')
                _, loss, acc, g_acc = sess.run(
                    [train_op, avg_loss, accuracy, g_accuracy],
                    feed_dict={images: train_images,
                               labels: train_labels})
                print("pass=%d, batch=%d, loss=%f, acc=%f" %
                      (pass_id, batch_id, loss, acc))
                iters += 1
                if iters > args.num_skip_batch:
                    num_samples += len(data)
                if args.iterations == iters:
                    break

        duration = time.clock() - start_time
        imgs_per_sec = num_samples / duration
        print("duration=%fs, performance=%fimgs/s" % (duration, imgs_per_sec))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    print_arguments()
    run_benchmark()
