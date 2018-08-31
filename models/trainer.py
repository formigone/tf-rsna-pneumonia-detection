from __future__ import division

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import itertools
import time
import os
import matplotlib.pyplot as plt

from data import gen_input


def parse_args():
    flags = argparse.ArgumentParser()
    flags.add_argument('--train_set', type=str, help='TFRecord used for training')
    flags.add_argument('--val_set', type=str, help='TFRecord used for evaluation')
    flags.add_argument('--model_dir', type=str, help='Path to saved_model')

    flags.add_argument('--cluster_spec', type=str, help='Distributed training stuff')

    flags.add_argument('--mode', type=str, default='train', help='Either train or predict.')
    flags.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    flags.add_argument('--dropout_keep_prob', type=float, default=0.5, help='The probability that each element is kept.')
    flags.add_argument('--norm_2', type=bool, default=False, help='When set, input is normalized with (W / 255 + 0.5) instead of (W / 255).')
    flags.add_argument('--spatial_squeeze', type=bool, default=True, help='If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C], where B is batch_size and C is number of classes.')
    flags.add_argument('--augment', type=int, default=1, help='Whether the training data should be augmented with random flips and other visual adjustments.')
    flags.add_argument('--num_classes', type=int, default=1000, help='Number of classes to classify.')
    flags.add_argument('--batch_size', type=int, default=16, help='Input function batch size.')
    flags.add_argument('--buffer_size', type=int, default=64, help='Input function buffer size.')
    flags.add_argument('--img_width', type=int, default=299, help='Width of input image.')
    flags.add_argument('--img_height', type=int, default=299, help='Height of input image.')

    # map_first=False, num_parallel_calls=None, prefetch=0
    flags.add_argument('--map_first', type=bool, default=False, help='If input pipeline should perform map before shuffle and repeat')
    flags.add_argument('--num_parallel_calls', type=int, default=None, help='Input pipeline optimization')
    flags.add_argument('--prefetch', type=int, default=0, help='Input pipeline optimization')
    flags.add_argument('--throttle_secs', type=int, default=900, help='Do not re-evaluate unless the last evaluation was started at least this many seconds ago.')

    flags.add_argument('--predict_set', type=str, help='TFRecord used for prediction.')
    flags.add_argument('--predict_csv', type=str, help='Name of test file.')
    flags.add_argument('--max_steps', type=int, default=10000, help='How many steps to take during training - including current step.')
    flags.add_argument('--epochs', type=int, default=1, help='How many epochs before the input_fn throws OutOfRange exception.')

    args = flags.parse_args()
    return args


def run(model_fn):
    args = parse_args()
    tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))

    model_params = {
        'train_set': args.train_set,
        'val_set': args.val_set,
        'model_dir': args.model_dir,
        'learning_rate': args.learning_rate,
        'dropout_keep_prob': args.dropout_keep_prob,
        'spatial_squeeze': args.spatial_squeeze,
        'num_classes': args.num_classes,
        'width': args.img_width,
        'height': args.img_height,
    }

    tf.logging.debug('Args: {}'.format(args))
    tf.logging.debug('Model Params: {}'.format(model_params))

    if args.cluster_spec:
        os.environ['TF_CONFIG'] = args.cluster_spec

    estimator = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=model_fn, params=model_params)

    if args.mode == 'train':
        train_input_fn = gen_input(args.train_set.split(','),
                                   batch_size=args.batch_size,
                                   buffer_size=args.buffer_size,
                                   img_shape=[args.img_height, args.img_width, 3],
                                   augment=bool(args.augment),
                                   norm_2=args.norm_2,
                                   map_first=args.map_first,
                                   num_parallel_calls=args.num_parallel_calls,
                                   prefetch=args.prefetch,
                                   repeat=args.epochs)
        eval_input_fn = gen_input(args.val_set.split(','),
                                  batch_size=args.batch_size,
                                  buffer_size=args.buffer_size,
                                  norm_2=args.norm_2,
                                  img_shape=[args.img_height, args.img_width, 3])

        tf.logging.info('Training for {}'.format(None if args.max_steps < 1 else args.max_steps))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None if args.max_steps < 1 else args.max_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, throttle_secs=args.throttle_secs)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif args.mode == 'predict':
        input_fn = gen_input(args.predict_set.split(','),
                                  batch_size=args.batch_size,
                                  buffer_size=args.buffer_size,
                                  norm_2=args.norm_2,
                                  img_shape=[args.img_height, args.img_width, 3])

        tf.logging.debug('Generating predictions using {}'.format(args.predict_set))
        predictions = estimator.predict(input_fn=input_fn)
        tf.logging.debug('Got predictions')
        tf.logging.debug('Fields: has pneumonia | male | age | box_x_min_0 | box_y_min_0 | box_width_0 | box_height_0 | box_x_min_1 | box_y_min_1 | box_width_1 | box_height_1 | box_x_min_2 | box_y_min_2 | box_width_2 | box_height_2 | box_x_min_3 | box_y_min_3 | box_width_3 | box_height_3')

        # for pred in predictions:
        #   print('{}'.format(pred['cholesterol']))
        # return

        tf.logging.debug('Reading input csv {}'.format(args.predict_csv))
        df = pd.read_csv(args.predict_set)
        # out_fh.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
        for pred, [_, row] in itertools.izip(predictions, df.iterrows()):
            print('{},{} {} {} {} {}'.format(
                row['patientId'].replace('data/stage-1-train-raw/', '').replace('.jpeg', ''),
                pred['predictions'][0],
                pred['predictions'][3] * row[-2],
                pred['predictions'][4] * row[-1],
                pred['predictions'][5] * row[-2],
                pred['predictions'][6] * row[-1],
            ))
    elif args.mode == 'weights':
        weights = estimator.get_variable_value('Conv2d_1a_7x7/weights')
        path = 'weights-vis-bn/{}'.format(int(time.time()))
        os.makedirs(path)

        for index in range(64):
            w = np.array(weights[:, :, :, index])
            w = ((w - w.min()) * (1 / (w.max() - w.min()) * 255)).astype('uint8')
            plt.imsave('{}/weights-{}.png'.format(path, index), w)
    else:
        raise ValueError('Invalid mode')
