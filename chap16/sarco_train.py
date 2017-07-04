import argparse
import logging
import numpy as np
import pandas as pd
import signal
import tensorflow as tf

from util import read_all_sarcos_data

# What we want to predict
TARGET_COLUMN = 'y1'

# Model parameters
HIDDEN1_UNITS = 70
HIDDEN2_UNITS = 50
L1_REGULARIZATION = 1.0
L2_REGULARIZATION = 14.0

# Tuning parameters for training
BATCH_SIZE = 1024
LEARNING_RATE = 0.01

# How often to output summaries
CHECKPOINT_INTERVAL_SECS = 60
SUMMARY_INTERVAL_STEPS = 500

def create_feature_columns(data):
    mean = data.mean()
    std = data.std()
    def create_feature_column(column_name):
        return tf.feature_column.numeric_column(
            column_name,
            normalizer_fn=lambda x: (x - mean[column_name])/std[column_name])
    feature_columns_list = [create_feature_column(column) for column in data.columns if column.startswith('x')]
    return feature_columns_list

def _weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 2)
    return tf.Variable(
        initial_value=initial, name=name,
        trainable=True,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])

def _bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.2)
    return tf.Variable(
        initial_value=initial, name=name,
        trainable=True,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES])

def inference(input, hidden1_units, hidden2_units):
    with tf.name_scope('hidden1'):
        weights = _weight_variable((input.shape[1].value, hidden1_units,), name='weights')
        biases = _bias_variable((hidden1_units,), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = _weight_variable((hidden1_units, hidden2_units,), name='weights')
        biases = _bias_variable((hidden2_units,), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('prediction'):
        weights = _weight_variable((hidden2_units, 1,), name='weights')
        biases = _bias_variable((1,), name='biases')
        prediction = tf.matmul(hidden2, weights) + biases
    return tf.squeeze(prediction)

def loss(prediction, actual, l1_regularization, l2_regularization):
    # Actual loss that we can compare models against
    mse = tf.losses.mean_squared_error(labels=actual, predictions=prediction, reduction=tf.losses.Reduction.SUM)
    # Include regularization in the loss function but don't return it.
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_regularization)
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regularization)
    regularizer = tf.contrib.layers.sum_regularizer([l1_regularizer, l2_regularizer])
    tf.contrib.layers.apply_regularization(regularizer=regularizer)
    return mse

def train(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.train.create_global_step()    
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def input_fn(data, batch_size):
    return tf.estimator.inputs.pandas_input_fn(
        x=data,
        y=data[TARGET_COLUMN],
        batch_size=batch_size,
        shuffle=True,
        num_epochs=None)()

def input_layer(features, feature_columns):
    return tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)

def _configure_logger(log_level):
    numeric_log_level = getattr(logging, log_level, None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: {:s}'.format(log_level))
    tf.logging.set_verbosity(numeric_log_level)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Sarcos model.')

    parser.add_argument(
        '--restart',
        action='store_const',
        const=True,
        default=False,
        help='Whether to start the training process over. By default, training continues from the last checkpoint.')

    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/sarco/logs',
        help='Where to put model checkpoints and summaries.')

    parser.add_argument(
        '--log_level',
        type=str,
        default='WARN',
        help='At what level to log at.')    

    FLAGS, _ = parser.parse_known_args()

    _configure_logger(FLAGS.log_level)

    if FLAGS.restart and tf.gfile.Exists(FLAGS.log_dir):
        tf.logging.info('Deleting old model.')
        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    training_data, test_data = read_all_sarcos_data()
    tf.logging.info('Data successfully loaded.')

    tf.logging.info('Building graph.')
    graph = tf.Graph()
    with graph.as_default():
        # Ensure reproducibility while debugging.
        tf.set_random_seed(2019)

        # Intialize the various layers of the graph.
    
        # Input layer to get batches of features
        features, target = input_fn(training_data, BATCH_SIZE)

        # Decide how to process the features into model input.
        feature_columns = create_feature_columns(pd.concat([training_data, test_data]))
        input = input_layer(features, feature_columns)

        # Do prediction/inference on the a training batch.
        prediction = inference(input, HIDDEN1_UNITS, HIDDEN2_UNITS)
        
        # Loss function to minimize. Remember to include regularization.
        mse = loss(prediction, target, L1_REGULARIZATION, L2_REGULARIZATION)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        # Minimizer that will create a global step.
        train_op = train(total_loss, LEARNING_RATE)

        # Summarize training run periodically
        tf.summary.scalar(name='loss', tensor=total_loss)
        # mse isn't normalized to making training more efficient
        tf.summary.scalar(name='mse', tensor=tf.divide(mse, BATCH_SIZE))
        
        # Let the user control whether to end the session.
        session_should_end = False
        def end_session(signum, stack_frame):
            global session_should_end
            session_should_end = True
        signal.signal(signal.SIGINT, end_session)
        signal.signal(signal.SIGTERM, end_session)

        tf.logging.info('Starting a session. Logging to {:s}'.format(FLAGS.log_dir))
        tf.logging.info('Press CTRL + C to stop training.')

        scaffold = tf.train.Scaffold(saver=tf.train.Saver(
            sharded=True,
            max_to_keep=1000,   # Effectively keep all models.
            allow_empty=True))
        sess = tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            save_checkpoint_secs=CHECKPOINT_INTERVAL_SECS,
            save_summaries_steps=SUMMARY_INTERVAL_STEPS,
            log_step_count_steps=SUMMARY_INTERVAL_STEPS,
            checkpoint_dir=FLAGS.log_dir)
        while sess.should_stop() is False:
            sess.run(train_op)
            if session_should_end:
                tf.logging.info('Ending training session.')
                sess.close()
