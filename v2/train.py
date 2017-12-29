import gzip
import argparse
import time

import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import inception_resnet_v2

DATA_DIM = 3 * 224 * 224  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
CLASS_DIM = 102


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='resnet',
        help='The model for image classification',
        choices=[
            'alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet',
            'inception-resnet-v2'
        ])
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='cifar10',
        choices=['cifar10', 'flowers'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--step', type=int, default=100, help='The number of iterations showing a loss.')
    parser.add_argument(
        '--skip_batch_num', type=int, default=20, help='The first num of minibatch num to skip, for better performance test')
    parser.add_argument(
        '--iterations', type=int, default=120, help='The number of final iteration.')

    args = parser.parse_args()

    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

    if args.data_set == "cifar10":
        CLASS_DIM = 10
        DATA_DIM = [3, 32, 32]
    elif args.data_set == "flowers":
        CLASS_DIM = 102
        DATA_DIM = [3, 224, 224]

    # PaddlePaddle init
    paddle.init(use_gpu=args.device == 'GPU', trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    extra_layers = None
    learning_rate = 0.01
    if args.model == 'alexnet':
        out = alexnet.alexnet(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg13':
        out = vgg.vgg13(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg16':
        out = vgg.vgg16(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg19':
        out = vgg.vgg19(image, class_dim=CLASS_DIM)
    elif args.model == 'resnet':
        out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
        learning_rate = 0.1
    elif args.model == 'googlenet':
        out, out1, out2 = googlenet.googlenet(image, class_dim=CLASS_DIM)
        loss1 = paddle.layer.cross_entropy_cost(
            input=out1, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out1, label=lbl)
        loss2 = paddle.layer.cross_entropy_cost(
            input=out2, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out2, label=lbl)
        extra_layers = [loss1, loss2]
    elif args.model == 'inception-resnet-v2':
        assert DATA_DIM == 3 * 331 * 331 or DATA_DIM == 3 * 299 * 299
        out = inception_resnet_v2.inception_resnet_v2(
            image, class_dim=CLASS_DIM, dropout_rate=0.5, data_dim=DATA_DIM)

    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        learning_rate=learning_rate / args.batch_size, )
    # learning_rate_decay_a=0.1,
    # learning_rate_decay_b=128000 * 35,
    # learning_rate_schedule="discexp", 
    # regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
    #                                                  args.batch_size),)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.cifar.train10()
            if args.data_set == 'cifar10' else paddle.dataset.flowers.train(),
            buf_size=5120),
        batch_size=args.batch_size)
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10()
        if args.data_set == 'cifar10' else paddle.dataset.flowers.test(),
        batch_size=args.batch_size)

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 extra_layers=extra_layers)

    class NameSpace:
        pass
    ns = NameSpace()
    ns.pass_start = time.time()
    ns.batch_start = time.time()
    ns.start = time.time()
    ns.batch_size = args.batch_size
    ns.log_step = args.step
    ns.iterator = 0
    ns.end_iter = args.iterations
    ns.skip_batch_num = args.skip_batch_num

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):

            if event.batch_id % ns.log_step == 0:
                batch_end = time.time()
                print "\nPass %d, Batch %d, Cost %f, %s, elapse:%f" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    (batch_end - ns.batch_start))
                ns.batch_start = time.time()

            if ns.iterator == ns.skip_batch_num:
                ns.start = time.time()
            if ns.iterator == ns.end_iter:
                duration = time.time() - ns.start
                im_num = (ns.iterator - ns.skip_batch_num) * ns.batch_size
                examples_per_sec = im_num / duration
                sec_per_batch = duration / (ns.iterator - args.skip_batch_num)

                print('\nTotal examples: %d, total time: %.5f' %
                      (im_num, duration))
                print('%.5f examples/sec, %.5f sec/batch \n' %
                      (examples_per_sec, sec_per_batch))

            ns.iterator += 1
        if isinstance(event, paddle.event.EndPass):
            pass_end = time.time()
            # with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            #     trainer.save_parameter_to_tar(f)
            # result = trainer.test(reader=test_reader)
            # print "\nTest with Pass %d, %s,elapse:%f" % (
            #    event.pass_id, result.metrics, (pass_end - ns.pass_start))
            print "\nTest with Pass %d,elapse:%f" % (event.pass_id,
                                                     (pass_end - ns.pass_start))
            ns.pass_start = time.time()
            exit(1)

    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
