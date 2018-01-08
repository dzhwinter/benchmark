from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.profiler as profiler
from paddle.v2.fluid.layer_helper import LayerHelper


def parse_args():
    parser = argparse.ArgumentParser("LSTM model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--stacked_num', type=int, default=2, help='Stacked LSTM Layer num.')
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
        action='store_true',
        help='If set, use nvprof for CUDA.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def lstm(x, c_pre_init, hidden_dim, forget_bias=None):
    """
    This function helps create an operator for the LSTM (Long Short Term
    Memory) cell that can be used inside an RNN.
    """
    helper = LayerHelper('lstm_unit', **locals())
    rnn = fluid.layers.StaticRNN()
    with rnn.step():
        c_pre = rnn.memory(init=c_pre_init)
        x_t = rnn.step_input(x)

        before_fc = fluid.layers.concat(input=[x_t, c_pre], axis=1)
        after_fc = fluid.layers.fc(input=before_fc, size=hidden_dim * 4)

        dtype = x.dtype
        c = helper.create_tmp_variable(dtype)
        h = helper.create_tmp_variable(dtype)

        helper.append_op(
            type='lstm_unit',
            inputs={"X": after_fc,
                    "C_prev": c_pre},
            outputs={"C": c,
                     "H": h},
            attrs={"forget_bias": forget_bias})

        rnn.update_memory(c_pre, c)
        rnn.output(h)

    return rnn()


def lstm_model(data, dict_dim, class_dim=2):
    batch_size = args.batch_size
    emb_dim = args.emb_dim
    seq_len = args.seq_len
    stacked_num = args.stacked_num

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    emb = fluid.layers.reshape(x=emb, shape=[batch_size, seq_len, emb_dim])
    emb = fluid.layers.transpose(x=emb, axis=[1, 0, 2])

    c_pre_init = fluid.layers.fill_constant(
        dtype=emb.dtype, shape=[batch_size, emb_dim], value=0.0)
    c_pre_init.stop_gradient = False
    layer_1_out = lstm(emb, c_pre_init=c_pre_init, hidden_dim=emb_dim)
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


def chop_data(data, chop_len, batch_size):
    data = [(list(x[0] + [0] * chop_len)[:chop_len], x[1]) for x in data]

    return data[:batch_size]


def prepare_feed_data(data, place):
    tensor_words = to_lodtensor(map(lambda x: x[0], data), place)

    label = np.array(map(lambda x: x[1], data)).astype("int64")
    label = label.reshape([-1, 1])
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
        dtype="int64",
        lod_level=1)
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
            paddle.dataset.imdb.train(word_dict),
            buf_size=25000),  # only for speed
        batch_size=args.batch_size)
    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    iters = 0
    for pass_id in xrange(args.pass_num):
        accuracy.reset(exe)
        for batch_id, data in enumerate(train_reader()):

            chopped_data = chop_data(
                data, chop_len=args.seq_len, batch_size=args.batch_size)
            tensor_words, tensor_label = prepare_feed_data(chopped_data, place)

            loss, acc = exe.run(
                fluid.default_main_program(),
                feed={"words": tensor_words,
                      "label": tensor_label},
                fetch_list=[avg_cost] + accuracy.metrics)
            pass_acc = accuracy.eval(exe)

            print("pass=%d, batch=%d, iters=%d ,loss=%f, acc=%f, pass_acc=%f" %
                  (pass_id, batch_id, iters, loss, acc, pass_acc))

            iters += 1
            if iters == args.iterations:
                return


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            run_benchmark(lstm_model, args)
    else:
        run_benchmark(lstm_model, args)
