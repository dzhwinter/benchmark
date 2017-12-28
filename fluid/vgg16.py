"""VGG16 benchmark in Fluid"""
from __future__ import print_function

import sys
import time
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import argparse
import functools

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--batch_size', type=int, default=32, help="Batch size for training.")
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
    default='NCHW',
    choices=['NCHW', 'NHWC'],
    help='The data order, now only support NCHW.')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    choices=['cifar10', 'flowers'],
    help='Optional dataset for benchmark.')

parser.add_argument(
    '--skip_batch_num',
    type=int,
    default=20,
    help='The first num of minibatch num to skip, for better performance test')
parser.add_argument(
    '--iterations',
    type=int,
    default=120,
    help='The number of final iteration.')
parser.add_argument(
    '--step', type=int, default=100, help='The number of iterations showing a loss.')
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
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


def main():
    if args.data_set == "cifar10":
        classdim = 10
        data_shape = [3, 32, 32]
    else:
        classdim = 102
        data_shape = [3, 224, 224]

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = vgg16_bn_drop(images)
    predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    opts = optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

    # inference program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_accuracy = fluid.evaluator.Accuracy(
            input=predict, label=label, main_program=inference_program)
        test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
        inference_program = fluid.io.get_inference_program(
            test_target, main_program=inference_program)

    # data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)

    # test
    def test(exe):
        test_accuracy.reset(exe)
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            exe.run(inference_program,
                    feed={"pixel": img_data,
                          "label": y_data},
                    fetch_list=[avg_cost] + test_accuracy.metrics)

        return test_accuracy.eval(exe)

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    iters = 0
    for pass_id in range(args.num_passes):
        # train     
        if iters == args.iterations:
            break
        accuracy.reset(exe)   
        pass_start_time = time.time()
        batch_start_time = time.time()
        start_time = time.time()
        num_samples = 0
        for batch_id, data in enumerate(train_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(data_shape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            outs = exe.run(fluid.default_main_program(),
                                feed={"pixel": img_data,
                                      "label": y_data},
                                fetch_list=[avg_cost] + accuracy.metrics if batch_id % args.step==0 else [])
            if batch_id % args.step == 0:
                batch_end_time = time.time()
                pass_acc = accuracy.eval(exe)
                print(
                    "Pass_id:%d, batch_id:%d, Iter: %d, loss: %.5f, acc: %.5f, pass_acc: %.5f, elapse: %f"
                    % (pass_id, batch_id, iters, outs[0][0], outs[1][0],
                       pass_acc[0], (batch_end_time - batch_start_time)))
                batch_start_time = time.time()

            num_samples += len(data)
            if iters == args.skip_batch_num:
                start_time = time.time()
            if iters == args.iterations:
                break
            iters += 1

        pass_elapsed = time.time() - pass_start_time
        print("Iter: %d, elapse: %f" % (iters, pass_elapsed))
        duration = time.time() - start_time
        examples_per_sec = num_samples / duration
        sec_per_batch = duration / (iters - args.skip_batch_num)
        print('\nTotal examples: %d, total time: %.5f' % (num_samples, duration))
        print('%.5f examples/sec, %.5f sec/batch \n' %
              (examples_per_sec, sec_per_batch))

        pass_test_acc = test(exe)
        print(
            "Pass = %d, Training performance = %f imgs/s, Test accuracy = %f\n"
            % (pass_id, num_samples / pass_elapsed, pass_test_acc))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print_arguments()
    main()
