"""VGG16 benchmark in Fluid"""
from __future__ import print_function

import sys
import time
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import argparse
import functools

parser = argparse.ArgumentParser("VGG16 benchmark.")
parser.add_argument(
    '--batch_size', type=int, default=128, help="Batch size for training.")
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help="Learning rate for training.")
parser.add_argument('--num_passes', type=int, default=50, help="No. of passes.")
parser.add_argument(
    '--device',
    type=str,
    default='GPU',
    choices=['CPU', 'GPU'],
    help="The device type.")
parser.add_argument(
    '--data_format',
    type=str,
    default='NHWC',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
args = parser.parse_args()


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=10, act=None)
    return fc2


def eval_test(exe):
    test_reader = paddle.batch(paddle.dataset.cifar.test10(), batch_size=100)
    accuracy.reset(exe)
    for batch_id, data in enumerate(test_reader()):
        img_data = np.array(map(lambda x: x[0].reshape([3, 32, 32]),
                                data)).astype(DTYPE)
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([len(y_data), 1])

        exe.run(framework.default_main_program(),
                feed={"pixel": img_data,
                      "label": y_data},
                fetch_list=[avg_cost] + accuracy.metrics)

    pass_acc = accuracy.eval(exe)
    return pass_acc


def main():
    classdim = 10
    data_shape = [3, 32, 32]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = vgg16_bn_drop(images)
    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    opts = optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10(), buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(paddle.dataset.cifar.test10(), batch_size=100)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.GPUPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    iters = 0
    for pass_id in range(args.num_passes):
        # train
        start_time = time.clock()
        num_samples = 0
        accuracy.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            batch_size = 1
            for i in y_data.shape:
                batch_size = batch_size * i
            y_data = y_data.reshape([batch_size, 1])

            loss, acc = exe.run(fluid.default_main_program(),
                                feed={"pixel": img_data,
                                      "label": y_data},
                                fetch_list=[avg_cost] + accuracy.metrics)
            iters += 1
            num_samples += len(data)
            print("Pass = %d, Iters = %d, Loss = %f, Accuracy = %f" %
                  (pass_id, iters, loss, acc))

        pass_elapsed = time.clock() - start_time
        # test
        accuracy.reset(exe)
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([len(y_data), 1])

            exe.run(fluid.default_main_program(),
                    feed={"pixel": img_data,
                          "label": y_data},
                    fetch_list=[avg_cost] + accuracy.metrics)

        pass_acc = accuracy.eval(exe)
        print(
            "Pass = %d, Training performance = %f imgs/s, Test accuracy = %f\n"
            % (pass_id, num_samples / pass_elapsed, pass_acc))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print_arguments()
    main()
