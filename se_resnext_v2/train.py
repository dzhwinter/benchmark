import math
import numpy as np
import os
import sys
import time
import argparse

import paddle
import paddle.dataset.flowers as flowers
import paddle.fluid as fluid
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import paddle.fluid.profiler as profiler

from model import SE_ResNeXt, lenet
fluid.default_startup_program().random_seed = 100


def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    args = parser.parse_args()
    return args


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()
    with init_on_cpu():
        epoch = fluid.layers.floor(global_step / step_each_epoch)
        lr = learning_rate / 2.
        decayed_lr = lr * (fluid.layers.cos(epoch * (math.pi / epochs)) + 1)
    return decayed_lr


def train_parallel_exe(learning_rate,
                       batch_size,
                       num_passes,
                       lr_strategy=None,
                       layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float64')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    ret = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    out = ret[-1]
    # out = lenet(input=image, class_dim=class_dim)
    # out = resnet_imagenet(input=image, class_dim=class_dim, layers=layers)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    opts = optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        flowers.train(
            use_xmap=False, mapper=flowers.test_mapper, buffered_size=1),
        batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
    fetch_list = [avg_cost, acc_top1, acc_top5]

    dshape = [3, 224, 224]
    args = parse_args()
    Iter = 0
    for pass_id in range(num_passes):
        train_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            image_data = np.array(map(lambda x: x[0].reshape(dshape),
                                      data)).astype('float64')
            label_data = np.array(map(lambda x: x[1], data)).astype('int64')
            label_data = label_data.reshape([-1, 1])
            ret_numpy = exe.run(
                fluid.default_main_program(),
                feed={'image': image_data,
                      'label': label_data},
                fetch_list=fetch_list)
            loss = ret_numpy.pop(0)
            acc1 = ret_numpy.pop(0)
            acc5 = ret_numpy.pop(0)

            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)

            if batch_id % 1 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4}"
                      .format(pass_id, batch_id, loss, acc1, acc5))
                sys.stdout.flush()
            Iter += 1
            if Iter == args.iterations:
                # if batch_id == 50:
                exit(0)


if __name__ == '__main__':
    lr_strategy = None
    method = train_parallel_exe
    method(
        learning_rate=0.1,
        batch_size=16,
        num_passes=5,
        lr_strategy=lr_strategy,
        layers=50)
