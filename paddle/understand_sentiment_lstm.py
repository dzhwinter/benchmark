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
        '--emb_dim', type=int, default=32, help='The embedding dim.')
    parser.add_argument(
        '--seq_len',
        type=int,
        default=80,
        help='The sequence length of one sentence.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
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


def lstm_model(data, dict_dim, class_dim=2):
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    seq_len = args.seq_len

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.reshape(x=emb, shape=[batch_size, seq_len, args.emb_dim])
    emb = fluid.layers.transpose(x=emb, axis=[1, 0, 2])

    c_pre_init = fluid.layers.fill_constant(
        dtype=emb.dtype, shape=[batch_size, emb_dim], value=0.0)
    layer_1_out = fluid.layers.lstm(
        emb, c_pre_init=c_pre_init, hidden_dim=emb_dim)
    layer_1_out = fluid.layers.transpose(x=layer_1_out, axis=[1, 0, 2])

    prediction = fluid.layers.fc(input=layer_1_out,
                                 size=class_dim,
                                 act="softmax")

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


def chop_data(data, chop_len=80, batch_size=50):
    data = [(x[0][:chop_len], x[1]) for x in data if len(x[0]) >= chop_len]

    return data[:batch_size]


def prepare_feed_data(data, place):
    tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

    label = np.array(map(lambda x: x[1], data)).astype("int64")
    label = label.reshape([len(label), 1])
    tensor_label = fluid.LoDTensor()
    tensor_label.set(label, place)

    return tensor_words, tensor_label


def run_benchmark(model, args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()
    start_time = time.time()
    word_dict = paddle.dataset.imdb.word_dict()

    print("load word dict successfully")

    dict_dim = len(word_dict)
    data = fluid.layers.data(
        name="words",
        shape=[args.seq_len * args.batch_size, 1],
        append_batch_size=False,
        dtype="int64")
    label = fluid.layers.data(
        name="label",
        shape=[args.batch_size, 1],
        append_batch_size=False,
        dtype="int64")
    prediction = model(data, dict_dim)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.002)
    adam_optimizer.minimize(avg_cost)
    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=args.batch_size * 10),
        batch_size=args.batch_size)
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.GPUPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for it, pass_id in enumerate(xrange(args.pass_num)):
        accuracy.reset(exe)
        if iter == args.iterations:
            break
        for data in train_reader():
            chopped_data = chop_data(data)
            tensor_words, tensor_label = prepare_feed_data(chopped_data, place)

            loss, acc = exe.run(
                fluid.default_main_program(),
                feed={"words": tensor_words,
                      "label": tensor_label},
                fetch_list=[cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)
            print("Iter: %d, loss: %s, acc: %s, pass_acc: %s" %
                  (iter, str(loss), str(acc), str(pass_acc)))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            run_benchmark(lstm_model, args)
    else:
        run_benchmark(lstm_model, args)
