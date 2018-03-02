import os
import math
import random
import cPickle
import functools
import numpy as np
import paddle.v2 as paddle
from PIL import Image, ImageEnhance

random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 10240

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.BILINEAR)
    return img


def Scale(img, size):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)


def CenterCrop(img, size):
    w, h = img.size
    th, tw = int(size), int(size)
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def random_crop_ycx(img, size):
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.08, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x1 = random.randint(0, img.size[0] - w)
            y1 = random.randint(0, img.size[1] - h)

            img = img.crop((x1, y1, x1 + w, y1 + h))
            assert (img.size == (w, h))

            return img.resize((size, size), Image.BILINEAR)

    img = Scale(img, size)
    img = CenterCrop(img, size)
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                             scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.size[0] - w)
    j = random.randint(0, img.size[1] - h)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image_imagepath2(sample, mode):
    img = np.random.rand(3, 224, 224)
    img -= img_mean
    img /= img_std

    lab = np.random.randint(0, 999)

    if mode == 'train' or mode == 'test':
        return img, lab
    elif mode == 'infer':
        return img


def fake_reader():
    img, lab = process_image_imagepath2(None, "train")
    while True:
        yield img, lab


def process_image_imagepath(sample, mode, color_jitter, rotate):
    imgpath = sample[0]
    img = Image.open(imgpath)
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = random_crop_ycx(img, DATA_DIM)
    else:
        img = Scale(img, 256)
        img = CenterCrop(img, DATA_DIM)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'test':
        return img, sample[1]
    elif mode == 'infer':
        return img


def _reader_creator_imagepath(data,
                              mode,
                              shuffle=False,
                              color_jitter=False,
                              rotate=False):
    def reader():
        index = range(0, len(data['image']))
        if shuffle:
            random.shuffle(index)
        for idx in index:
            if mode == 'train' or mode == 'test':
                yield data['image'][idx], data['label'][idx]
            elif mode == 'infer':
                yield [data['image'][idx]]

    mapper = functools.partial(process_image_imagepath2, mode=mode)
    #mapper = functools.partial(
    #    process_image_imagepath, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def _reader_creator(data, mode, shuffle=False, color_jitter=False,
                    rotate=False):
    def reader():
        index = range(0, len(data['image']))
        if shuffle:
            random.shuffle(index)
        for idx in index:
            if mode == 'train' or mode == 'test':
                yield data['image'][idx], data['label'][idx]
            elif mode == 'infer':
                yield [data['image'][idx]]

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train():
    return fake_reader


def test():
    return fake_reader
