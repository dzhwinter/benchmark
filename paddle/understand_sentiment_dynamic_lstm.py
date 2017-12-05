from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.profiler as profiler


def parse_args():
    parser = argparse.ArgumentParser("LSTM model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--stacked_num', type=int, default=3, help='Stacked LSTM Layer size.')
    parser.add_argument(
        '--emb_dim', type=int, default=32, help='The embedding dim.')
    parser.add_argument(
        '--hid_dim',
        type=int,
        default=32,
        help='The sequence length of one sentence.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--infer_only', action='store_true', help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof', action='store_true', help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_false',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def dynamic_lstm_model(data, dict_dim, class_dim=2):
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    hid_dim = args.hid_dim
    stacked_num = args.stacked_num

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    inputs = emb

    for i in range(stacked_num):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(input=fc, size=hid_dim)
        inputs = lstm
    lstm_last = fluid.layers.sequence_pool(input=inputs, pool_type='max')

    prediction = fluid.layers.fc(input=[lstm_last],
                                 size=class_dim,
                                 act='softmax')

    return prediction


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def run_benchmark(model, args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()
    start_time = time.time()
    word_dict = paddle.dataset.imdb.word_dict()

    print("load word dict successfully")

    dict_dim = len(word_dict)

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    prediction = model(data, dict_dim)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.002)
    adam_optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict),
            buf_size=args.batch_size * 10),
        batch_size=args.batch_size)
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.GPUPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for it, pass_id in enumerate(xrange(args.pass_num)):
        accuracy.reset(exe)
        if iter == args.iterations:
            break
        for data in train_reader():
            tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([args.batch_size, 1])

            tensor_label = fluid.LoDTensor()
            tensor_label.set(label, place)

            loss, acc = exe.run(
                fluid.default_main_program(),
                feed={"words": tensor_words,
                      "label": tensor_label},
                fetch_list=[avg_cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)
            print("Iter: %d, loss: %s, acc: %s, pass_acc: %s" %
                  (it, str(loss), str(acc), str(pass_acc)))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            run_benchmark(dynamic_lstm_model, args)
    else:
        run_benchmark(dynamic_lstm_model, args)
