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

from models import resnet
import paddle
import paddle.fluid as fluid
# import paddle.dataset.flowers as flowers
import paddle.fluid.profiler as profiler

fluid.default_startup_program().random_seed = 111


def parse_args():
    parser = argparse.ArgumentParser('ResNet-50 parallel profile.')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--batch_size_per_gpu', type=int, default=12, help='')
    parser.add_argument(
        '--use_mem_opt',
        type=distutils.util.strtobool,
        default=True,
        help='use memory optimize or not.')
    parser.add_argument(
        '--do_profile',
        type=distutils.util.strtobool,
        default=False,
        help='do profile or not.')
    parser.add_argument('--number_iteration', type=int, default=150, help='')
    parser.add_argument('--pass_num', type=int, default=10, help='')
    parser.add_argument('--display_step', type=int, default=10, help='')
    parser.add_argument('--skip_first_steps', type=int, default=30, help='')
    parser.add_argument('--warmup', type=int, default=20, help='')
    parser.add_argument(
        '--fix_data_in_gpu',
        type=distutils.util.strtobool,
        default=True,
        help='')
    parser.add_argument(
        '--use_recordio',
        type=distutils.util.strtobool,
        default=False,
        help='.')
    parser.add_argument(
        '--balance_parameter_opt_between_cards',
        type=distutils.util.strtobool,
        default=False,
        help='balance parameter opt between cards')
    parser.add_argument(
        '--show_record_time',
        type=distutils.util.strtobool,
        default=False,
        help='')

    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s=%s' % (arg, value))


def fake_reader():
    while True:
        img = np.random.rand(3, 224, 224)
        lab = np.random.randint(0, 101)
        yield img, lab


def train():
    return fake_reader


def generate_recordio(data_shape, data_set_iterator, output_file, batch_size=1):
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        reader = paddle.batch(data_set_iterator(), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        feed_list=[
            fluid.layers.data(
                name='data', shape=data_shape, dtype='float32'),
            fluid.layers.data(
                name='label', shape=[1], dtype='int64'),
        ],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_file(output_file, reader,
                                                          feeder)


def net_conf(image, label, class_dim):
    out = resnet.resnet_imagenet(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return out, avg_cost


def add_optimizer(args, avg_cost):
    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    optimizer.minimize(avg_cost)

    if args.use_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())


def train_parallel_exe(args):
    class_dim = 102
    image_shape = [3, 224, 224]

    if args.use_recordio:
        recordio_name = './flowers_1.recordio'
        if not os.path.exists(recordio_name):
            data_set_iterator = paddle.dataset.flowers.train
            print("generate {0} ... ".format(recordio_name))
            generate_recordio(image_shape, data_set_iterator, recordio_name)

        file_list = [recordio_name] * 8
        data_file = fluid.layers.io.open_files(
            filenames=file_list,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            thread_num=4,
            pass_num=args.pass_num)
        data_file = fluid.layers.io.batch(
            data_file, batch_size=args.batch_size_per_gpu)
        data_file = fluid.layers.io.double_buffer(data_file)
        image, label = fluid.layers.io.read_file(data_file)
    else:
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    prediction, avg_cost = net_conf(image, label, class_dim)
    add_optimizer(args, avg_cost)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce if args.balance_parameter_opt_between_cards else fluid.BuildStrategy.ReduceStrategy.AllReduce

    exe = fluid.ParallelExecutor(
        loss_name=avg_cost.name,
        use_cuda=True,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader = feeder.decorate_reader(
        paddle.batch(
            train(), batch_size=args.batch_size_per_gpu),
        multi_devices=True)

    train_reader_iter = train_reader()
    if args.fix_data_in_gpu:
        data = train_reader_iter.next()
        feed_data = data

    # warm up
    for batch_id in xrange(args.warmup):
        exe.run([],
                feed=feed_data
                if args.fix_data_in_gpu else train_reader_iter.next())

    time_record = []
    train_start = time.time()
    img_count = 0
    for batch_id in xrange(args.number_iteration):
        if args.do_profile and batch_id >= 5 and batch_id < 8:
            with profiler.profiler('All', 'total',
                                   '/tmp/profile_parallel_exe') as prof:
                if args.use_recordio:
                    exe.run([])
                else:
                    exe.run([],
                            feed=feed_data if args.fix_data_in_gpu else
                            train_reader_iter.next())
            continue

        if args.use_recordio:
            cost_val = exe.run([avg_cost.name] if (batch_id + 1) %
                               args.display_step == 0 else [])
        else:
            cost_val = exe.run(
                [avg_cost.name]
                if (batch_id + 1) % args.display_step == 0 else [],
                feed=feed_data
                if args.fix_data_in_gpu else train_reader_iter.next())

        img_count += args.batch_size

        if (batch_id + 1) % args.display_step == 0:
            train_stop = time.time()
            step_time = train_stop - train_start
            time_record.append(step_time)

            print("iter=%d, cost=%s, elapse=%f, img/sec=%f" %
                  ((batch_id + 1), np.array(cost_val[0]), step_time,
                   img_count / step_time))

            img_count = 0
            train_start = time.time()

    skip_time_record = args.skip_first_steps / args.display_step
    time_record[0:skip_time_record] = []

    if args.show_record_time:
        for i, ele in enumerate(time_record):
            print("iter:{0}, time consume:{1}".format(i, ele))

    img_count = (
        args.number_iteration - args.skip_first_steps) * args.batch_size

    print("average time:{0}, img/sec:{1}".format(
        np.mean(time_record), img_count / np.sum(time_record)))


if __name__ == '__main__':
    args = parse_args()

    cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    cards_num = len(cards.split(","))
    args.batch_size = args.batch_size_per_gpu * cards_num

    print_arguments(args)
    print("cards_num=" + str(cards_num))

    train_parallel_exe(args)
