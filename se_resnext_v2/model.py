import math
import numpy as np
import os
import sys
import time

# import paddle.v2 as paddle
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter


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
        bias_attr=False,
        use_cudnn=False)
    # return conv
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    ### initializer parameter
    # print >> sys.stderr, "pool shape:", pool.shape
    stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    # print >> sys.stderr, "squeeze shape:", squeeze.shape
    stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
    excitation = fluid.layers.fc(input=squeeze, size=num_channels, act='relu')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut_old(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


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
    scale = conv2
    # scale = squeeze_excitation(
    #     input=conv2,
    #     num_channels=num_filters * 2,
    #     reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def resnet_imagenet(input, class_dim, infer=False, layers=50):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, pool1, 64, stages[0], 1)
    res2 = layer_warp(block_func, res1, 128, stages[1], 2)
    res3 = layer_warp(block_func, res2, 256, stages[2], 2)
    res4 = layer_warp(block_func, res3, 512, stages[3], 2)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return Out


def SE_ResNeXt(input, class_dim, infer=False, layers=50):
    # supported_layers = [50, 152]
    # if layers not in supported_layers:
    #     print("supported layers are", supported_layers, \
    #           "but input layer is ", layers)
    #     exit()
    # if layers == 50:
    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]

    ret = []
    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=7, stride=2, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='avg')

    # block = 3
    # conv = bottleneck_block(
    #     input=conv,
    #     num_filters=num_filters[block],
    #     stride=2,
    #     cardinality=cardinality,
    #     reduction_ratio=reduction_ratio)

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=7, pool_type='avg', global_pooling=True)
    # if not infer:
    #     drop = fluid.layers.dropout(x=pool, dropout_prob=0.5, seed=1)
    # else:
    #     drop = pool
    drop = pool
    # print >> sys.stderr, "drop shape:", drop.shape
    stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    ret.append(out)
    return ret


def lenet(input, class_dim, infer=False):
    conv1 = fluid.layers.conv2d(input, 32, 5, 1, act=None, use_cudnn=False)
    # conv1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(conv1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 50, 5, 1, act=None, use_cudnn=False)
    # conv2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(conv2, 2, 'max', 2)
    fc1 = fluid.layers.fc(pool2, size=500, act='relu')
    fc2 = fluid.layers.fc(fc1, size=10, act='softmax')
    return fc2
