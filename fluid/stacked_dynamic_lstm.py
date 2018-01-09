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
        '--batch_size',
        type=int,
        default=32,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--stacked_num',
        type=int,
        default=5,
        help='Number of lstm layers to stack. (default: %(default)d)')
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=512,
        help='Dimension of embedding table. (default: %(default)d)')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help='Hidden size of lstm unit. (default: %(default)d)')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=100,
        help='Epoch number to train. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002,
        help='Learning rate used to train. (default: %(default)f)')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
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


def dynamic_lstm_model(dict_size,
                       embedding_dim,
                       hidden_dim,
                       stacked_num,
                       class_num=2,
                       is_train=True):
    word_idx = fluid.layers.data(
        name="word_idx", shape=[1], dtype="int64", lod_level=1)
    embedding = fluid.layers.embedding(
        input=word_idx, size=[dict_size, embedding_dim])

    sentence = fluid.layers.fc(input=embedding, size=hidden_dim * 4, bias_attr=True)

    # input = embedding
    # for i in range(stacked_num):
    #     fc = fluid.layers.fc(input=input, size=hidden_dim * 4, bias_attr=True)
    #     hidden, cell = fluid.layers.dynamic_lstm(
    #         input=fc, size=hidden_dim * 4, use_peepholes=False)
    #     input = hidden

    # lstm_out = fluid.layers.sequence_pool(input=input, pool_type='max')
    # prediction = fluid.layers.fc(input=lstm_out, size=class_num, act='softmax')
    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        hidden_dim = 32
        prev_hidden = rnn.memory(value=0.0, shape=[hidden_dim])
        prev_cell = rnn.memory(value=0.0, shape=[hidden_dim])

        def gate_common(
                ipt,
                hidden,
                size, ):
            gate0 = fluid.layers.fc(input=ipt, size=size, bias_attr=True)
            gate1 = fluid.layers.fc(input=hidden, size=size, bias_attr=False)
            gate = fluid.layers.sums(input=[gate0, gate1])
            return gate

        forget_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, hidden_dim))
        input_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, hidden_dim))
        output_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, hidden_dim))
        cell_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, hidden_dim))

        cell = fluid.layers.sums(input=[
            fluid.layers.elementwise_mul(
                x=forget_gate, y=prev_cell), fluid.layers.elementwise_mul(
                    x=input_gate, y=cell_gate)
        ])

        hidden = fluid.layers.elementwise_mul(
            x=output_gate, y=fluid.layers.tanh(x=cell))

        rnn.update_memory(prev_cell, cell)
        rnn.update_memory(prev_hidden, hidden)
        rnn.output(hidden)

    lstm_out = fluid.layers.sequence_pool(input=rnn(), pool_type='max')
    prediction = fluid.layers.fc(input=lstm_out, size=class_num, act='softmax')

    if not is_train: return word_idx, prediction

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return (word_idx, label), prediction, label, avg_cost


def train(args):
    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

    word_dict = paddle.dataset.imdb.word_dict()
    dict_size = len(word_dict)

    feeding_list, prediction, label, avg_cost = dynamic_lstm_model(
        dict_size, args.embedding_dim, args.hidden_dim, args.stacked_num)

    adam_optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    adam_optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    # clone from default main program
    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)
        test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
        inference_program = fluid.io.get_inference_program(test_target)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=25000),
        batch_size=args.batch_size)

    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.test(word_dict), buf_size=25000),
        batch_size=args.batch_size)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=feeding_list, place=place)
    exe.run(fluid.default_startup_program())

    def do_validation():
        test_accuracy.reset(exe)

        for data in test_reader():
            loss, acc = exe.run(inference_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + test_accuracy.metrics)

        return test_accuracy.eval(exe)

    for pass_id in xrange(args.pass_num):
        pass_start_time = time.time()
        words_seen = 0
        accuracy.reset(exe)
        for batch_id, data in enumerate(train_reader()):
            words_seen += sum([len(seq[0]) for seq in data])

            loss, acc = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost] + accuracy.metrics)
            train_acc = accuracy.eval(exe)

            print("pass_id=%d, batch_id=%d, loss: %f, acc: %f, avg_acc: %f" %
                  (pass_id, batch_id, loss, acc, train_acc))

        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        test_acc = do_validation()
        print("pass_id=%d, test_acc: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_acc, words_per_sec, time_consumed))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    if args.infer_only:
        pass
    else:
        if args.use_nvprof and args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            train(args)
