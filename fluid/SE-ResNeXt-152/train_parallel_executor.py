#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse
import distutils.util
import numpy as np

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.v2.dataset.flowers as flowers
import paddle.fluid.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser('SE-ResNeXt-152 parallel-executor model.')
    parser.add_argument(
        '--use_mem_opt',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize')
    parser.add_argument('--per_gpu_batch_size', type=int, default=12, help='')
    parser.add_argument(
        '--number_iteration',
        type=int,
        default=10,
        help='total batch num for per_gpu_batch_size')

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s=%s' % (arg, value))


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)
    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False, layers=152):
    supported_layers = [50, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, "but input layer is ",
              layers)
        exit()
    if layers == 50:
        cardinality = 32
        reduction_ratio = 16
        depth = [3, 4, 6, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    elif layers == 152:
        cardinality = 64
        reduction_ratio = 16
        depth = [3, 8, 36, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=2, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def net_conf(image, label, class_dim):
    out = SE_ResNeXt152(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    #accuracy = fluid.evaluator.Accuracy(input=out, label=label)
    #accuracy5 = fluid.evaluator.Accuracy(input=out, label=label, k=5)
    accuracy = fluid.layers.accuracy(input=out, label=label)
    accuracy5 = fluid.layers.accuracy(input=out, label=label, k=5)
    return out, avg_cost, accuracy, accuracy5


def train():
    args = parse_args()

    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    batch_size = args.per_gpu_batch_size * cards_num

    print_arguments(args)
    print("cards_num=" + str(cards_num))
    print("batch_size=" + str(batch_size))

    class_dim = 1000
    image_shape = [3, 224, 224]

    main = fluid.Program()
    startup = fluid.Program()

    with fluid.program_guard(main, startup):
        data_file = fluid.layers.open_recordio_file(
            filename='./resnet_152.recordio_batch_size_12_3_224_224',  #  ./resnet_152.recordio_batch_size_2
            shapes=[[-1, 3, 224, 224], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'])
        image, label = fluid.layers.read_file(data_file)

        prediction, avg_cost, accuracy, accuracy5 = net_conf(image, label,
                                                             class_dim)

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=[100], values=[0.1, 0.2]),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
        opts = optimizer.minimize(avg_cost)

        if args.use_mem_opt:
            fluid.memory_optimize(fluid.default_main_program())

        exe = fluid.ParallelExecutor(loss_name=avg_cost.name, use_cuda=True)

        batch_id = 0
        time_record = []
        # with profiler.profiler('All', 'total', '/tmp/profile') as prof:
        for i in xrange(args.number_iteration):
            t1 = time.time()
            exe.run([avg_cost.name] if batch_id % 10 == 0 else [])
            t2 = time.time()
            period = t2 - t1
            time_record.append(period)

            if batch_id % 10 == 0:
                print("trainbatch {0},  time{1}".format(batch_id,
                                                        "%2.2f sec" % period))
            batch_id += 1

        del time_record[0]
        for ele in time_record:
            print ele

        print("average time:{0}".format(np.mean(time_record)))


if __name__ == '__main__':
    train()
