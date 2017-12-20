"""seq2seq model for fluid."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time
import distutils.util

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.framework as framework
from paddle.v2.fluid.param_attr import ParamAttr
from paddle.v2.fluid.executor import Executor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--word_vector_dim",
    type=int,
    default=512,
    help="The dimension of embedding table. (default: %(default)d)")
parser.add_argument(
    "--encoder_size",
    type=int,
    default=512,
    help="The size of encoder bi-rnn unit. (default: %(default)d)")
parser.add_argument(
    "--decoder_size",
    type=int,
    default=512,
    help="The size of decoder rnn unit. (default: %(default)d)")
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="The sequence number of a batch data. (default: %(default)d)")
parser.add_argument(
    "--dict_size",
    type=int,
    default=30000,
    help="The dictionary capacity. Dictionaries of source sequence and "
    "target dictionary have same capacity. (default: %(default)d)")
parser.add_argument(
    "--pass_number",
    type=int,
    default=2,
    help="The pass number to train. (default: %(default)d)")
parser.add_argument(
    "--mode",
    type=str,
    default='train',
    choices=['train', 'infer'],
    help="Do training or inference. (default: %(default)s)")
parser.add_argument(
    "--beam_size",
    type=int,
    default=3,
    help="The width for beam searching. (default: %(default)d)")
parser.add_argument(
    "--use_gpu",
    type=distutils.util.strtobool,
    default=True,
    help="Whether use gpu. (default: %(default)d)")
parser.add_argument(
    "--max_length",
    type=int,
    default=250,
    help="The max length of sequence when doing generation. "
    "(default: %(default)d)")


def seq_to_seq_net(word_vector_dim,
                   encoder_size,
                   decoder_size,
                   source_dict_dim,
                   target_dict_dim,
                   is_generating=False,
                   beam_size=3,
                   max_length=250):
    """Construct a seq2seq network."""
    feeding_list = ["source_sequence", "target_sequence", "label_sequence"]

    def bi_lstm_encoder(input_seq, size):
        input_forward_proj = fluid.layers.fc(input=input_seq,
                                             size=size * 4,
                                             act='tanh')
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=size * 4)
        input_reversed_proj = fluid.layers.fc(input=input_seq,
                                              size=size * 4,
                                              act='tanh')
        reversed, _ = fluid.layers.dynamic_lstm(
            input=input_reversed_proj, size=size * 4, is_reverse=True)
        return forward, reversed

    src_word_idx = fluid.layers.data(
        name=feeding_list[0], shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, word_vector_dim],
        dtype='float32')

    src_forward, src_reversed = bi_lstm_encoder(
        input_seq=src_embedding, size=encoder_size)

    encoded_vector = fluid.layers.concat(
        input=[src_forward, src_reversed], axis=1)

    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    backward_first = fluid.layers.sequence_pool(
        input=src_reversed, pool_type='first')

    decoder_boot = fluid.layers.fc(input=backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act='tanh')

    def lstm_decoder_with_attention(target_embedding, encoder_vec, encoder_proj,
                                    decoder_boot, decoder_size):
        def simple_attention(encoder_vec, encoder_proj, decoder_state):
            decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                                 size=decoder_size)
            decoder_state_expand = fluid.layers.sequence_expand(
                x=decoder_state_proj, y=encoder_proj)
            concated = fluid.layers.concat(
                input=[decoder_state_expand, encoder_proj], axis=1)
            attention_weights = fluid.layers.fc(input=concated,
                                                size=1,
                                                bias_attr=False)
            attention_weights = fluid.layers.sequence_softmax(
                x=attention_weights)
            weigths_reshape = fluid.layers.reshape(
                x=attention_weights, shape=[-1])
            scaled = fluid.layers.elementwise_mul(
                x=encoder_vec, y=weigths_reshape, axis=0)
            context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
            return context

        rnn = fluid.layers.DynamicRNN()

        cell_init = fluid.layers.fill_constant_batch_size_like(
            input=decoder_boot,
            value=0.0,
            shape=[-1, decoder_size],
            dtype='float32')
        cell_init.stop_gradient = False

        with rnn.block():
            current_word = rnn.step_input(target_embedding)
            hidden_mem = rnn.memory(init=decoder_boot)
            cell_mem = rnn.memory(init=cell_init)
            context = simple_attention(encoder_vec, encoder_proj, hidden_mem)
            decoder_inputs = fluid.layers.concat(
                input=[context, current_word], axis=1)
            h, c = fluid.layers.lstm_unit(
                x_t=decoder_inputs,
                hidden_t_prev=hidden_mem,
                cell_t_prev=cell_mem)
            rnn.update_memory(hidden_mem, h)
            rnn.update_memory(cell_mem, c)
            out = fluid.layers.fc(input=h,
                                  size=target_dict_dim,
                                  bias_attr=ParamAttr(),
                                  act='softmax')
            rnn.output(out)

        return rnn()

    if not is_generating:
        trg_word_idx = fluid.layers.data(
            name=feeding_list[1], shape=[1], dtype='int64', lod_level=1)

        trg_embedding = fluid.layers.embedding(
            input=trg_word_idx,
            size=[target_dict_dim, word_vector_dim],
            dtype='float32')

        prediction = lstm_decoder_with_attention(trg_embedding, encoded_vector,
                                                 encoded_proj, decoder_boot,
                                                 decoder_size)

        label = fluid.layers.data(
            name=feeding_list[2], shape=[1], dtype='int64', lod_level=1)

        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        return avg_cost, feeding_list


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    lod_t = core.LoDTensor()
    lod_t.set(flattened_data, place)
    lod_t.set_lod([lod])
    return lod_t


def train():
    avg_cost, feeding_list = seq_to_seq_net(
        args.word_vector_dim,
        args.encoder_size,
        args.decoder_size,
        args.dict_size,
        args.dict_size,
        False,
        beam_size=args.beam_size,
        max_length=args.max_length)

    optimizer = fluid.optimizer.Adam(learning_rate=5e-5)
    optimizer.minimize(avg_cost)

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(args.dict_size), buf_size=1000),
        batch_size=args.batch_size)

    place = core.GPUPlace() if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    for pass_id in xrange(args.pass_number):
        for batch_id, data in enumerate(train_batch_generator()):
            src_seq = to_lodtensor(map(lambda x: x[0], data), place)
            trg_seq = to_lodtensor(map(lambda x: x[1], data), place)
            lbl_seq = to_lodtensor(map(lambda x: x[2], data), place)

            fetch_outs = exe.run(
                framework.default_main_program(),
                feed=dict(zip(*[feeding_list, (src_seq, trg_seq, lbl_seq)])),
                fetch_list=[avg_cost])

            avg_cost_val = np.array(fetch_outs[0])

            print('pass_id=%d, batch=%d, avg_cost=%f' %
                  (pass_id, batch_id, avg_cost_val))


def infer():
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        infer()
