import argparse
import cPickle
import os
import random

import numpy
import paddle.v2.dataset.imdb as imdb
import paddle.v2.fluid as fluid
from paddle.v2 import batch

try:
    with open('word_dict.pkl', 'r') as f:
        word_dict = cPickle.load(f)
except:
    word_dict = imdb.word_dict()
    with open('word_dict.pkl', 'w') as f:
        cPickle.dump(word_dict, f, cPickle.HIGHEST_PROTOCOL)


def cache_reader(reader, clean):
    print 'Reading data to memory'
    fn = 'data.pkl'
    if clean:
        try:
            os.remove(fn)
        except:
            pass
    try:
        with open(fn, 'r') as f:
            items = cPickle.load(f)
    except:
        items = list(reader())
        with open(fn, 'w') as f:
            cPickle.dump(items, f, cPickle.HIGHEST_PROTOCOL)

    print 'Done. data size %d' % len(items)

    def __impl__():
        offsets = range(len(items))
        random.shuffle(offsets)
        for i in offsets:
            yield items[i]

    return __impl__


def crop_sentence(reader, crop_size):
    unk_value = word_dict['<unk>']

    def __impl__():
        for item in reader():
            if len([x for x in item[0] if x != unk_value]) < crop_size:
                yield item

    return __impl__


def main():
    args = parse_args()
    data = fluid.layers.data(
        name="words", shape=[1], lod_level=1, dtype='int64')
    sentence = fluid.layers.embedding(
        input=data, size=[len(word_dict), args.emb_dim])

    sentence = fluid.layers.fc(input=sentence, size=200, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        lstm_size = 32
        prev_hidden = rnn.memory(value=0.0, shape=[lstm_size])
        prev_cell = rnn.memory(value=0.0, shape=[lstm_size])

        def gate_common(
                ipt,
                hidden,
                size, ):
            gate0 = fluid.layers.fc(input=ipt, size=size, bias_attr=True)
            gate1 = fluid.layers.fc(input=hidden, size=size, bias_attr=False)
            gate = fluid.layers.sums(input=[gate0, gate1])
            return gate

        forget_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        input_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        output_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))
        cell_gate = fluid.layers.sigmoid(
            x=gate_common(word, prev_hidden, lstm_size))

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

    last = fluid.layers.sequence_pool(rnn(), 'last')
    logit = fluid.layers.fc(input=last, size=2, act='softmax')
    loss = fluid.layers.cross_entropy(
        input=logit,
        label=fluid.layers.data(
            name='label', shape=[1], dtype='int64'))
    loss = fluid.layers.mean(x=loss)

    adam = fluid.optimizer.Adam()
    adam.minimize(loss)

    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def train_loop(pass_num, crop_size):
        cache = cache_reader(
            crop_sentence(imdb.train(word_dict), crop_size), clean=args.clean)
        for pass_id in range(pass_num):
            train_reader = batch(cache, batch_size=args.batch_size)
            for batch_id, data in enumerate(train_reader()):
                tensor_words = to_lodtensor([x[0] for x in data], place)
                label = numpy.array([x[1] for x in data]).astype("int64")
                label = label.reshape((-1, 1))
                loss_np = exe.run(fluid.default_main_program(),
                                  feed={"words": tensor_words,
                                        "label": label},
                                  fetch_list=[loss])[0]
                print 'Pass', pass_id, 'Batch', batch_id, 'loss', loss_np
            print 'Pass', pass_id, 'Done'

    train_loop(args.pass_num, args.crop_size)


def parse_args():
    parser = argparse.ArgumentParser("Understand Sentiment by Dynamic RNN.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=int(os.environ.get('BATCH_SIZE', '64')),
        help='The minibatch size.')
    parser.add_argument(
        '--emb_dim',
        type=int,
        default=int(os.environ.get('EMB_DIM', '32')),
        help='The embedding dim.')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=int(os.environ.get('PASS_NUM', '100')),
        help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--crop_size',
        type=int,
        default=int(os.environ.get('CROP_SIZE', '1500')),
        help='The max sentence length of input. Since this model use plain RNN,'
        ' Gradient could be explored if sentence is too long')
    parser.add_argument(
        '--clean',
        type=bool,
        default=os.environ.get('CLEAN', 'False').lower() != 'false',
        help='clean the cached pickle file.')
    args = parser.parse_args()
    return args


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = numpy.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


if __name__ == '__main__':
    main()
