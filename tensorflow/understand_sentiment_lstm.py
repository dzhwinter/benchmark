from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

import tensorflow as tf
from tensorflow.contrib import rnn
import paddle.v2 as paddle

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size.""")
tf.app.flags.DEFINE_integer('emb_dim', 32, "The embedding dim.")
tf.app.flags.DEFINE_integer('seq_len', 80,
                            "The sequence length of one sentence.")
tf.app.flags.DEFINE_integer('iterations', 35, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('pass_num', 100, "The number of passes.")
tf.app.flags.DEFINE_boolean('infer_only', False,
                            """Only run the forward pass.""")


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(args.iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def lstm_model(data, dict_dim, class_dim=2):
    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    seq_len = FLAGS.seq_len

    with tf.name_scope("lstm") as scope:
        embedding = tf.Variable(tf.truncated_normal([dict_dim, emb_dim]))

        # NOTE(dzhwinter) : paddle dynamic_lstm(lstm_op) do not have peepholes

        lstm_input = tf.nn.embedding_lookup(embedding, data)

        lstm_input = tf.unstack(lstm_input, seq_len, 1)
        # lstm_cell = rnn.BasicLSTMCell(emb_dim, forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=emb_dim, use_peepholes=False)
        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * emb_dim)

        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # outputs, state = tf.nn.dynamic_rnn(
        #     lstm_cell, lstm_input, initial_state=initial_state, dtype=tf.float32)
        outputs, states = rnn.static_rnn(
            lstm_cell, lstm_input, dtype=tf.float32)
        last_output = outputs[-1]

        fc_weights = tf.Variable(
            tf.truncated_normal([emb_dim, class_dim]), dtype=tf.float32)
        bias = tf.Variable(
            tf.constant(
                value=0.0, shape=[class_dim], dtype=tf.float32))

        prediction = tf.matmul(last_output, fc_weights) + bias

    return prediction


def padding_data(data, padding_size, value):
    data = data + [value] * padding_size
    return data[:padding_size]


def run_benchmark(model):
    start_time = time.time()

    word_dict = paddle.dataset.imdb.word_dict()
    print("load word dict successfully")
    dict_dim = len(word_dict)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict),
            buf_size=FLAGS.batch_size * 10),
        batch_size=FLAGS.batch_size)

    data = tf.placeholder(tf.int64, shape=[None, FLAGS.seq_len])
    label = tf.placeholder(tf.int64, shape=[None])
    prediction = model(data, dict_dim)
    cost = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(label, 2), logits=prediction)
    avg_cost = tf.reduce_mean(cost)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
    train_op = adam_optimizer.minimize(avg_cost)

    correct = tf.equal(tf.argmax(prediction, 1), label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    pass_accuracy = tf.metrics.accuracy(label, tf.argmax(prediction, axis=1))

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(config=config) as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_l)
        sess.run(init_g)
        for it in enumerate(xrange(FLAGS.pass_num)):
            if it == FLAGS.iterations:
                break
            for batch in train_reader():

                word_data = np.array(
                    map(lambda x: padding_data(x[0], FLAGS.seq_len, 0),
                        batch)).astype("int64")
                label_data = np.array(map(lambda x: x[1], batch)).astype(
                    "int64")

                _, loss, acc, pass_acc = sess.run(
                    [train_op, avg_cost, accuracy, pass_accuracy],
                    feed_dict={data: word_data,
                               label: label_data})
                print("Iter: %d, loss: %s, acc: %s, pass_acc: %s" %
                      (it, str(loss), str(acc), str(pass_acc)))


def main(_):
    args = dict(tf.flags.FLAGS.__flags)
    print_arguments(args)
    run_benchmark(lstm_model)


if __name__ == '__main__':
    tf.app.run()
