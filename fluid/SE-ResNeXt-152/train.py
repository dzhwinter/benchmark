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

import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import paddle.fluid.profiler as profiler

fluid.default_startup_program().random_seed = 111


def parse_args():
    parser = argparse.ArgumentParser('SE-ResNeXt-152 parallel profile.')
    parser.add_argument(
        '--class_number', type=int, default=1000, help='the class number')
    parser.add_argument(
        '--use_parallel_mode',
        type=str,
        default='parallel_exe',
        choices=['parallel_do', 'parallel_exe'],
        help='The parallel mode("parallel_do" or "parallel_exe").')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--per_gpu_batch_size', type=int, default=12, help='')
    parser.add_argument(
        '--use_mem_opt',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument(
        '--do_profile',
        type=distutils.util.strtobool,
        default=True,
        help='do profile or not.')
    parser.add_argument(
        '--number_iteration',
        type=int,
        default=50,
        help='total batch num for per_gpu_batch_size.')
    parser.add_argument('--display_step', type=int, default=1, help='')
    parser.add_argument(
        '--skip_first_steps',
        type=int,
        default=2,
        help='The first num of steps to skip, for better performance profile.')
    parser.add_argument(
        '--parallel',
        type=distutils.util.strtobool,
        default=True,
        help='It is valid only when parallel_mode is parallel_do.')
    parser.add_argument(
        '--use_nccl',
        type=distutils.util.strtobool,
        default=True,
        help='It is valid only when parallel_mode is parallel_do.')
    parser.add_argument(
        '--use_python_reader',
        type=distutils.util.strtobool,
        default=True,
        help='It is valid only when parallel_mode is parallel_do.'
        'If use_python_reader is True, python reader is used to feeding data,'
        'the process includes data transfer from CPU to GPU. Otherwise, '
        'the data which will be needed for training is in GPU side constantly.')

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    if args.use_parallel_mode == "parallel_do":
        for arg, value in sorted(vars(args).iteritems()):
            print('%s=%s' % (arg, value))
    else:
        args.use_nccl = True
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
    if ch_in != ch_out or stride != 1:
        filter_size = 1
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
    out = SE_ResNeXt(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    #accuracy = fluid.evaluator.Accuracy(input=out, label=label)
    #accuracy5 = fluid.evaluator.Accuracy(input=out, label=label, k=5)
    accuracy = fluid.layers.accuracy(input=out, label=label)
    accuracy5 = fluid.layers.accuracy(input=out, label=label, k=5)
    return out, avg_cost, accuracy, accuracy5


def add_optimizer(args, avg_cost):
    #optimizer = fluid.optimizer.SGD(learning_rate=0.002)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[100], values=[0.1, 0.2]),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    optimizer.minimize(avg_cost)

    if args.use_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())


def train_parallel_do(args):

    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if args.parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=args.use_nccl)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)
            out = SE_ResNeXt(input=image_, class_dim=class_dim)
            cost = fluid.layers.cross_entropy(input=out, label=label_)
            avg_cost = fluid.layers.mean(x=cost)
            accuracy = fluid.layers.accuracy(input=out, label=label_)
            pd.write_output(avg_cost)
            pd.write_output(accuracy)

        avg_cost, accuracy = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        accuracy = fluid.layers.mean(x=accuracy)
    else:
        out = SE_ResNeXt(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        accuracy = fluid.layers.accuracy(input=out, label=label)

    add_optimizer(args, avg_cost)

    place = fluid.CUDAPlace(0)
    # place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(flowers.train(), batch_size=args.batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader_iter = train_reader()
    if not args.use_python_reader:
        data = train_reader_iter.next()
        feed_dict = feeder.feed(data)

    time_record = []

    for batch_id in range(args.number_iteration):
        if args.do_profile and batch_id >= 5 and batch_id < 8:
            with profiler.profiler('All', 'total',
                                   '/tmp/profile_parallel_do') as prof:
                exe.run(fluid.default_main_program(),
                        feed=feeder.feed(train_reader_iter.next())
                        if args.use_python_reader else feed_dict,
                        fetch_list=[],
                        use_program_cache=True)
            continue

        train_start = time.time()
        cost_val = exe.run(fluid.default_main_program(),
                           feed=feeder.feed(train_reader_iter.next())
                           if args.use_python_reader else feed_dict,
                           fetch_list=[avg_cost.name]
                           if batch_id % args.display_step == 0 else [],
                           use_program_cache=True)
        train_stop = time.time()
        step_time = train_stop - train_start
        time_record.append(step_time)

        if batch_id % args.display_step == 0:
            print("iter=%d, elapse=%f, cost=%s" %
                  (batch_id, step_time, np.array(cost_val[0])))

    for _ in range(args.skip_first_steps):
        del time_record[0]

    for ele in time_record:
        print ele

    print("average time:{0}".format(np.mean(time_record)))


def train_parallel_exe(args):

    class_dim = 1000
    image_shape = [3, 224, 224]

    main = fluid.Program()
    startup = fluid.Program()

    with fluid.program_guard(main, startup):
        reader = fluid.layers.open_recordio_file(
            filename='./flowers_bs_12_3_224_224.recordio',
            shapes=[[-1, 3, 224, 224], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'])

        # currently, double buffer only supports one device.
        #data_file = fluid.layers.create_double_buffer_reader(reader=data_file, place='CUDA:0')
        image, label = fluid.layers.read_file(reader)

        prediction, avg_cost, accuracy, accuracy5 = net_conf(image, label,
                                                             class_dim)

        add_optimizer(args, avg_cost)

        exe = fluid.ParallelExecutor(
            loss_name=avg_cost.name, use_cuda=True, allow_op_delay=True)

        time_record = []

        for batch_id in xrange(args.number_iteration):

            if args.do_profile and batch_id >= 5 and batch_id < 8:
                with profiler.profiler('All', 'total',
                                       '/tmp/profile_parallel_exe') as prof:
                    exe.run([])
                continue

            t1 = time.time()
            cost_val = exe.run([avg_cost.name]
                               if batch_id % args.display_step == 0 else [])
            t2 = time.time()
            period = t2 - t1
            time_record.append(period)

            if batch_id % args.display_step == 0:
                print("iter=%d, elapse=%f, cost=%s" %
                      (batch_id, period, np.array(cost_val[0])))

        for _ in range(args.skip_first_steps):
            del time_record[0]

        for ele in time_record:
            print ele

        print("average time:{0}".format(np.mean(time_record)))


if __name__ == '__main__':
    args = parse_args()

    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    args.batch_size = args.per_gpu_batch_size * cards_num

    print_arguments(args)
    print("cards_num=" + str(cards_num))

    if args.use_parallel_mode == "parallel_do":
        train_parallel_do(args)
    else:
        train_parallel_exe(args)
