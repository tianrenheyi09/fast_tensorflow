#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from tensor_textcnn import TextCNN
# from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import jieba
import jieba.posseg as pseg
from tensor_birnn import TextBiRNN
# from text_birnn import TextBiRNN
from tensor_textdnn import TextDNN
from tensor_textrcnn import TextRCNN
from tensor_textrnn import TextRNN
from tensor_textfast import TextFast


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("model_type", "clf", "the type of model ,classify or regression(defalut=clf)")
tf.flags.DEFINE_string("using_nn_type", "textcnn", "the type of network(default:textcnn)")

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("train_data_file", "./data/cutclean_label_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "./data/vectors.bin", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_integer("hidden_size", 128, "Number of hiddern layer units(defalut:128)")
tf.flags.DEFINE_integer("hidden_layers", 2, "Number of hidden layers(default:2)")

tf.flags.DEFINE_integer("rnn_size", 300, "num of units rnn_size(default:3)")
tf.flags.DEFINE_integer("num_rnn_layers", 3, "number of rnn layers(default:3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_data(w2v_model=None):
    print("laoding data")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)
    max_document_length = max([len(x.split(" ")) for x in x_text])
    if (w2v_model == None):
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        vocab_size = len(vocab_processor.vocabulary_)
    else:
        x = data_helpers.get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)
        vocab_size = len(w2v_model.vocab_hash)
        print('use w2v .bin')

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, x_dev, y_train, y_dev, vocab_size


#####################--------------载入数据----
x_train, x_dev, y_train, y_dev, vocab_size = load_data(None)
#############--------定义nn类型
nn_type = "text_fasttext"


def train():
    # Training
    # ==================================================
    # x_train, x_dev, y_train, y_dev ,vocab_size= load_data(w2v_model)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if (nn_type == "text_cnn"):
                nn = TextCNN(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif nn_type == "text_birnn":
                nn = TextBiRNN(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    rnn_size=128,
                    num_layers=3,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif nn_type == "text_rnn":
                nn = TextRNN(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    rnn_size=128,
                    num_layers=3,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif nn_type == "text_rcnn":
                nn = TextBiRNN(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    rnn_size=128,
                    num_layers=3,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif nn_type == "text_dnn":
                nn = TextDNN(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    hidden_layes=2,
                    hidden_size=128,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif nn_type == "text_fasttext":
                nn = TextFast(
                    model_type=FLAGS.model_type,
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(nn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    nn.input_x: x_batch,
                    nn.input_y: y_batch,
                    nn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step,loss, accuracy = sess.run(
                    [train_op, global_step, nn.loss, nn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2],idx[:2]
                # train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    nn.input_x: x_batch,
                    nn.input_y: y_batch,
                    nn.dropout_keep_prob: 1.0
                }

                step,loss, accuracy = sess.run(
                    [global_step, nn.loss, nn.accuracy],
                    feed_dict)
                #
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # if writer:
                #     writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            def dev_test():
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    dev_step(x_batch_dev, y_batch_dev, writer=None)


            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                #
                # current_step = tf.train.global_step(sess, global_step)

            print("dev_test验证：")
            dev_test()




if __name__ == "__main__":
    # w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    train()
    attrs = []
    values = []
    for attr, value in sorted(FLAGS.__flags.items()):
        attrs += [attr]
        values += [value]
    info = pd.DataFrame()
    info["attr"] = attrs
    info["value"] = values
    info.to_csv(os.path.curdir + '/config.csv', index=False)

