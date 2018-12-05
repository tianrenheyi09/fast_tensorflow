#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
# from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import pandas as pd


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("valid_data_file", "./data/cutclean_label_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("w2v_file", "./data/vectors.bin", "w2v_file path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1543137628/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")


def load_data(w2v_model, max_document_length=1290):
    """Loads starter word-vectors and train/dev/test data."""
    # Load the starter word vectors
    print("Loading data...")
    x_text, y_test = data_helpers.load_data_and_labels(FLAGS.valid_data_file)
    y_test = np.argmax(y_test, axis=1)

    if (max_document_length == 0):
        max_document_length = max([len(x.split(" ")) for x in x_text])

    print('max_document_length = ', max_document_length)

    x = data_helpers.get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)

    return x, y_test


w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
w2v_model = w2v_wr.model
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print("cheach_point_file",checkpoint_file)
graph = tf.Graph()

sequence_length = 1290
num_classes = 2
input_x = tf.placeholder(tf.int32,shape=[None,sequence_length],name="input_x")
input_y = tf.placeholder(tf.int32,shape=[None,num_classes],name="input_y")
# dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
learning_rate = tf.placeholder(tf.float32,name="learning_rate")



with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        # input_x = graph.get_operation_by_name("input_x").outputs[0]
        # print("input_x ",sess.run(input_x))
        #
        #
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        print("dropout_keep_prob",sess.run(dropout_keep_prob))

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print("predictions :",sess.run(predictions))

        scores = graph.get_operation_by_name("output/scores").outputs[0]

        x_test, y_test = load_data(w2v_model, 1290)
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for index, x_test_batch in enumerate(batches):
            batch_predictions = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions[0]])

            probabilities = softmax(batch_predictions[1])

            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}".format(time_str, (index + 1) * FLAGS.batch_size))

# for x_test_batch in batches:
#                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
#                all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack(all_predictions)
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))

# predictions_human_readable1 = np.column_stack((np.array(x_test),
#                                              [int(prediction)+1 for prediction in all_predictions],
#                                              ["{}".format(probability) for probability in all_probabilities]))


predictions_human_readable1 = np.column_stack(([int(prediction) + 1 for prediction in all_predictions],
                                               ["{}".format(probability) for probability in all_probabilities]))

predict_results = pd.DataFrame(predictions_human_readable1, columns=['Label', 'Probabilities'])

print("Saving evaluation to {0}".format(out_path))
predict_results.to_csv(out_path, index=False)

#    with open(out_path, 'w') as f:
#        csv.writer(f).writerows(predictions_human_readable)



# if __name__ == "__main__":
#    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
#    eval(w2v_wr.model)



#
# def eval(w2v_model):
#    # Evaluation
#    # ==================================================
#    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
#    graph = tf.Graph()
#    with graph.as_default():
#        session_conf = tf.ConfigProto(
#          allow_soft_placement=FLAGS.allow_soft_placement,
#          log_device_placement=FLAGS.log_device_placement)
#        sess = tf.Session(config=session_conf)
#        with sess.as_default():
#            # Load the saved meta graph and restore variables
#            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#            saver.restore(sess, checkpoint_file)
#
#            # Get the placeholders from the graph by name
#            input_x = graph.get_operation_by_name("input_x").outputs[0]
#
#            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#            # Tensors we want to evaluate
#            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#
#            scores = graph.get_operation_by_name("output/scores").outputs[0]
#
#            x_test, y_test = load_data(w2v_model,1290)
#            # Generate batches for one epoch
#            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#
#            # Collect the predictions here
#            all_predictions = []
#            all_probabilities = None
#
#            for index,x_test_batch in enumerate(batches):
#                batch_predictions = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
#                all_predictions = np.concatenate([all_predictions, batch_predictions])
#
#                probabilities = softmax(batch_predictions[1])
#
#                if all_probabilities is not None:
#                    all_probabilities = np.concatenate([all_probabilities,probabilities])
#                else:
#                    all_probabilities = probabilities
#                time_str = datetime.datetime.now().isoformat()
#                print("{}: step {}".format(time_str, (index+1)*FLAGS.batch_size))
#
##            for x_test_batch in batches:
##                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
##                all_predictions = np.concatenate([all_predictions, batch_predictions])
#
#    # Print accuracy if y_test is defined
#    if y_test is not None:
#        correct_predictions = float(sum(all_predictions == y_test))
#        print("Total number of test examples: {}".format(len(y_test)))
#        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
#        print(metrics.classification_report(y_test, all_predictions))
#        print(metrics.confusion_matrix(y_test, all_predictions))
#
#    # Save the evaluation to a csv
#    predictions_human_readable = np.column_stack(all_predictions)
#    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
#    print("Saving evaluation to {0}".format(out_path))
#
#    predictions_human_readable = np.column_stack((np.array(x_test),
#                                                  [int(prediction)+1 for prediction in all_predictions],
#                                                  ["{}".format(probability) for probability in all_probabilities]))
#
#    predict_results = pd.DataFrame(predictions_human_readable, columns=['Content','Label','Probabilities'])
#
#    print("Saving evaluation to {0}".format(out_path))
#    predict_results.to_csv(out_path, index=False)
#
##    with open(out_path, 'w') as f:
##        csv.writer(f).writerows(predictions_human_readable)
#
#
#
# if __name__ == "__main__":
#    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
#    eval(w2v_wr.model)
