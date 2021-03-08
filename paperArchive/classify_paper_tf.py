# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial

import modeling
import optimization
import tokenization
import tensorflow as tf
import random
import pickle
import numpy as np
from predict_fast import FastPredict
from tf_parameters import FLAGS, InputExample,\
    PaddingInputExample, InputFeatures, DataProcessor

import os

class_label = ["单选题","多选题","判断题","填空题","解答题","复合题","其他", "完型填空"]
sys_train_file = "classify_train.xlsx"
sys_eval_file = "classify_eval.xlsx"

class PaperClassifyProcessor(DataProcessor):
    """Processor for good paper classifier"""

    def get_train_examples(self, data_dir):
        """See base class."""
        labels = self._read_excel(
            os.path.join(data_dir,
                         sys_train_file), 1)
        texts = self._read_excel(
            os.path.join(data_dir,
                         sys_train_file), 0)
        examples = []
        for (i, label) in enumerate(labels):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = texts[i]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        random.shuffle(examples)
        print('________train data len:___', len(examples))
        tf.logging.info(examples)
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        eval_path = os.path.join(data_dir, 'dataset_eval.pkl')
        examples_eval = []

        if not os.path.exists(eval_path):
            print("not existing pkl in the file, read from the xlsx.")
            labels = self._read_excel(
                os.path.join(data_dir,
                             sys_eval_file), 1)
            texts = self._read_excel(
                os.path.join(data_dir,
                             sys_eval_file), 0)
            for (i, label) in enumerate(labels):
                if i == 0:
                    continue
                guid = "dev-%d" % (i)
                text_a = texts[i]
                examples_eval.append(
                    InputExample(guid=guid, text_a=text_a, label=label))
            print('total eval length:', len(examples_eval))

            f = open(eval_path, 'wb')
            pickle.dump(examples_eval, f)
            f.close()
            print('________eval data len:', len(examples_eval))

        else:
            print("exist pkl in the path, load from pkl file...")
            f = open(eval_path, 'rb')
            examples_eval = pickle.load(f)
            f.close()
            print('________eval data len:', len(examples_eval))

        return examples_eval

    def get_labels(self, data_dir):
        """See base class."""
        return class_label

    def get_test_examples(self, data_dir):
        examples = []
        eval_path = os.path.join(data_dir, 'dataset_eval.pkl')
        if os.path.exists(eval_path):
            print('pkl existing........')
            f = open(eval_path, 'rb')
            examples = pickle.load(f)
            f.close()
            print('________test data len:', len(examples))

        return examples

    def get_batch_examples(self, texts):
        examples = []
        for (i, text) in enumerate(texts):
            guid = "predict-%d" % (i)
            text = tokenization.convert_to_unicode(text)
            label = "单选题"
            examples.append(
                InputExample(guid=guid, text_a=text, label=label))
        print('total batch length:', len(examples))
        print(examples)
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file=None):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([int(feature.label_id)])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids": tf.FixedLenFeature([], tf.float32),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # For regression
        # logits = tf.squeeze(logits)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        # For regression
        # loss_ = tf.square(tf.subtract(logits, labels))
        # loss = tf.reduce_mean(loss_)
        # per_example_loss = tf.reduce_mean(loss_)

        # For classification
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            # Replace TPU related code to GPU related
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     train_op=train_op,
            #     scaffold_fn=scaffold_fn)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions,
                    weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss,
                                       weights=is_real_example)
                # f1_score = tf.contrib.metrics.f1_score(
                #     label_ids,
                #     predictions)
                # auc = tf.metrics.auc(
                #     label_ids,
                #     predictions)
                # recall = tf.metrics.recall(
                #     label_ids,
                #     predictions)
                # precision = tf.metrics.precision(
                #     label_ids,
                #     predictions)
                # true_pos = tf.metrics.true_positives(
                #     label_ids,
                #     predictions)
                # true_neg = tf.metrics.true_negatives(
                #     label_ids,
                #     predictions)
                # false_pos = tf.metrics.false_positives(
                #     label_ids,
                #     predictions)
                # false_neg = tf.metrics.false_negatives(
                #     label_ids,
                #     predictions)
                return {
                    "eval_accuracy": accuracy,
                    # "f1_score": f1_score,
                    # "auc": auc,
                    # "precision": precision,
                    # "recall": recall,
                    # "true_positives": true_pos,
                    # "true_negatives": true_neg,
                    # "false_positives": false_pos,
                    # "false_negatives": false_neg,
                    "eval_loss": loss,
                }


            # Replace TPU realted code to GPU related.
            eval_metrics = metric_fn(per_example_loss, label_ids, logits,
                                     is_real_example)
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     # eval_metrics=eval_metrics,
            #     predictions={"logits": logits},
            #     scaffold_fn=scaffold_fn)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )

        else:
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     predictions={"logits": logits},
            #     scaffold_fn=scaffold_fn)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # # For regression
                # predictions={"logits": logits}
                predictions={"logits": logits, "probabilities": probabilities}
            )

        return output_spec

    return model_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features,seq_length,is_training,drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    # all_input_ids = []
    # all_input_mask = []
    # all_segment_ids = []
    # all_label_ids = []

    # for feature in features:
    #     all_input_ids.append(feature.input_ids)
    #     all_input_mask.append(feature.input_mask)
    #     all_segment_ids.append(feature.segment_ids)
    #     all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]
        #
        # num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        # d = tf.data.Dataset.from_tensor_slices({
        #     "input_ids":
        #         tf.constant(
        #             all_input_ids, shape=[num_examples, seq_length],
        #             dtype=tf.int32),
        #     "input_mask":
        #         tf.constant(
        #             all_input_mask,
        #             shape=[num_examples, seq_length],
        #             dtype=tf.int32),
        #     "segment_ids":
        #         tf.constant(
        #             all_segment_ids,
        #             shape=[num_examples, seq_length],
        #             dtype=tf.int32),
        #     "label_ids":
        #         tf.constant(all_label_ids, shape=[num_examples], dtype=tf.float32),
        # })
        #
        # if is_training:
        #     d = d.repeat()
        #     d = d.shuffle(buffer_size=100)
        #
        # d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        # output_shape=((num_examples, seq_length),)
        #d = tf.data.Dataset.from_generator(features,output_types=(tf.int32,
        # tf.int32,tf.int32,tf.float32))
        d = tf.data.Dataset.from_generator(features, {"input_ids": tf.int32,
                                                      "input_mask": tf.int32,
                                                      "segment_ids": tf.int32,
                                                      "label_ids": tf.int32},
                                           {"input_ids": [seq_length],
                                            "input_mask": [seq_length],
                                            "segment_ids": [seq_length],
                                            "label_ids": []})
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d=d.batch(1, drop_remainder=drop_remainder)
        feat=d.make_one_shot_iterator().get_next()

        return feat

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        # features.append(feature)

        # for fast-predict
        features.append({
            "input_ids":
                feature.input_ids,
            "input_mask":
                feature.input_mask,
            "segment_ids":
                feature.segment_ids,
            "label_ids":
                feature.label_id,
        })

    return features

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):

    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            # pdb.set_trace()
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items


    return 1.0 - numerator / denominator


class ClassifyPaperEngine(object):
    """Wapper of the the good sentence classification engine.
    """
    def __init__(self):
        self.FLAGS = FLAGS
        self.data_dir = './data/class_paper/'
        self.task_name = 'class_paper'
        self.vocab_file = 'multi_cased_L-12_H-768_A-12/vocab.txt'
        self.bert_config_file = 'multi_cased_L-12_H-768_A-12/bert_config.json'
        self.output_dir = './result/class_paper_base_v1.0/'
        self.init_checkpoint = 'multi_cased_L-12_H-768_A-12/bert_model.ckpt'
        self.do_train = True
        self.do_eval = True
        self.do_predict = False
        self.do_demo = False

        self.max_seq_length = 256
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.predict_batch_size = 4
        self.learning_rate = 1e-6
        self.num_train_epochs = 30

        # set train/eval file
        # if self.do_train:
        #     self.FLAGS.train_file = "classify_train.xlsx"
        # if self.do_eval:
        #     self.FLAGS.eval_file = "classify_eval.xlsx"

        tf.logging.set_verbosity(tf.logging.INFO)

        processors = {
            'class_paper': PaperClassifyProcessor
        }

        tokenization.validate_case_matches_checkpoint(self.FLAGS.do_lower_case,
                                                      self.init_checkpoint)

        if not self.do_train and not self.do_eval and not self.do_predict and not self.do_demo:
            raise ValueError(
                "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        if self.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(self.output_dir)

        task_name = self.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        self.processor = processors[task_name]()

        # label for regression
        self.label_list = self.processor.get_labels(self.data_dir)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.FLAGS.do_lower_case)

        # use auto increasement for gpu usage.
        tf.logging.info("load estimator...")
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options={"allow_growth": True}
        )

        run_config = tf.estimator.RunConfig(
            session_config=config,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.FLAGS.save_checkpoints_steps
        )

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None

        if self.do_train:
            train_examples = self.processor.get_train_examples(self.data_dir)
            num_train_steps = int(
                len(
                    train_examples) / self.train_batch_size * self.num_train_epochs)
            num_warmup_steps = int(num_train_steps * self.FLAGS.warmup_proportion)

        self.train_examples = train_examples
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.predict_drop_remainder = True if self.FLAGS.use_tpu else False

        self.model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(self.label_list),
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self.FLAGS.use_tpu,
            use_one_hot_embeddings=self.FLAGS.use_tpu)

        if self.do_train:
            batch_s = self.train_batch_size
        elif self.do_predict:
            batch_s = self.predict_batch_size
        elif self.do_eval:
            batch_s = self.eval_batch_size
        else:
            batch_s = self.predict_batch_size

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=run_config,
            params={"batch_size": batch_s}
        )

        self.fast_estimator = FastPredict(self.estimator, input_fn=partial(
            input_fn_builder, seq_length=self.max_seq_length,
            is_training=False, drop_remainder=False))

    def model_train(self):
        train_file = os.path.join(self.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            self.train_examples, self.label_list, self.max_seq_length,
            self.tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.train_batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True)
        self.estimator.train(input_fn=train_input_fn,
                             max_steps=self.num_train_steps)

    def model_eval(self):
        eval_examples = self.processor.get_dev_examples(self.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if self.FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % self.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(self.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, self.label_list, self.max_seq_length,
            self.tokenizer,
            eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", self.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if self.FLAGS.use_tpu:
            assert len(eval_examples) % self.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // self.eval_batch_size)

        eval_drop_remainder = True if self.FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = self.estimator.evaluate(input_fn=eval_input_fn,
                                         steps=eval_steps)

        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def model_predict(self):
        predict_examples = self.processor.get_test_examples(self.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if self.FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % self.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(self.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples,
                                                self.label_list,
                                                self.max_seq_length,
                                                self.tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.predict_batch_size)

        predict_drop_remainder = True if self.FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)
        # pdb.set_trace()
        output_predict_file = os.path.join(self.output_dir,
                                           "test_results.tsv")
        predicts, targets = [], []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1


    def fast_demo(self, texts):
        predict_examples = self.processor.get_batch_examples(texts)
        input_features = convert_examples_to_features(predict_examples,
                                                       self.label_list,
                                                       self.max_seq_length,
                                                       self.tokenizer)
        results = self.fast_estimator.predict(input_features)
        print(results)

        # import pdb; pdb.set_trace()

        predicts = [] # sentence score list
        for (i, result) in enumerate(results):
            # append type "good" logits
            # result only have one element, thus [0][0]
            prob = result["probabilities"].tolist()[0]
            # print(prob)
            predicts.append(class_label[prob.index(max(prob))])

        # print("predict result:", predicts)
        # if self.predict_batch_size == 1:
        #     print("batch_size= 1, final returns:", predicts)
        # else:
        #     print("batch_size > 1, final returns:", predicts)

        return predicts


def main(_):
    classModel = ClassifyPaperEngine()

    if classModel.do_demo:
       classModel.fast_demo(['I love u,I love u,I love u,I love u,I love u',
                            'I love u'])
    if classModel.do_train:
        classModel.model_train()

    if classModel.do_eval:
       classModel.model_eval()

    if classModel.do_predict:
       classModel.model_predict()

if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()
