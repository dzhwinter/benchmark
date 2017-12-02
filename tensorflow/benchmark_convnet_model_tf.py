"""
https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import paddle.v2 as paddle
#import paddle.v2.fluid.profiler as profiler
import tensorflow as tf

DTYPE = tf.float32


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet'],
        default='resnet',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=100, help='The number of passes.')
    parser.add_argument(
        '--order',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data order, now only support NCHW.')
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


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=0.9,
        epsilon=1e-05,
        center=True,
        scale=True,
        training=is_training,
        fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
    """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides,
                     data_format):
    """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
    """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut,
                      strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)


def resnet_generator(block_fn, layers, num_classes,
                     data_format='channels_last'):
    """
  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs,
            filters=64,
            kernel_size=7,
            strides=2,
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=3,
            strides=2,
            padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        inputs = block_layer(
            inputs=inputs,
            filters=64,
            block_fn=block_fn,
            blocks=layers[0],
            strides=1,
            is_training=is_training,
            name='block_layer1',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs,
            filters=128,
            block_fn=block_fn,
            blocks=layers[1],
            strides=2,
            is_training=is_training,
            name='block_layer2',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs,
            filters=256,
            block_fn=block_fn,
            blocks=layers[2],
            strides=2,
            is_training=is_training,
            name='block_layer3',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs,
            filters=512,
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            is_training=is_training,
            name='block_layer4',
            data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs,
            pool_size=7,
            strides=1,
            padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs,
                            [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def resnet(depth, class_dim):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {
            'block': building_block,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': building_block,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': bottleneck_block,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': bottleneck_block,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': bottleneck_block,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': bottleneck_block,
            'layers': [3, 24, 36, 3]
        }
    }

    if depth not in model_params:
        raise ValueError('Not a valid depth:', depth)

    params = model_params[depth]

    return resnet_generator(params['block'], params['layers'], class_dim)


def run_benchmark(batch_size, data_format='channels_last'):
    """Our model_fn for ResNet to be used with our Estimator."""

    class_dim = 102

    dshape = (None, 3, 224, 224) if data_format == 'channels_first' else (
        None, 224, 224, 3)

    images = tf.placeholder(DTYPE, shape=dshape)
    labels = tf.placeholder(tf.int64, shape=(None, ))
    one_hot_labels = tf.one_hot(labels, depth=class_dim)

    network = resnet(50, class_dim)
    logits = network(inputs=images, is_training=True)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels)
    avg_cost = tf.reduce_mean(cross_entropy)

    correct = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    g_accuracy = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=5120),
        batch_size=batch_size)

    with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

        for pass_id in range(args.pass_num):
            for batch_id, data in enumerate(train_reader()):
                images_data = np.array(
                    map(lambda x: np.transpose(x[0].reshape([3, 224, 224]), axes=[1, 2, 0]), data)).astype("float32")
                labels_data = np.array(map(lambda x: x[1], data)).astype(
                    'int64')
                _, loss, acc, g_acc = sess.run(
                    [train_op, avg_cost, accuracy, g_accuracy],
                    feed_dict={images: images_data,
                               labels: labels_data})
                print("pass=%d, batch=%d, loss=%f, acc=%f\n" %
                      (pass_id, batch_id, loss, acc))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.order == 'NHWC':
        raise ValueError('Only support NCHW order now.')
    if args.use_nvprof and args.device == 'GPU':
        #with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
        run_benchmark(args.batch_size)
    else:
        run_benchmark(args.batch_size)
