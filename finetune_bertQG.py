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
"""BERT finetuning runner for straight-forward QG
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pandas as pd
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .csv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 50,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_export_savedmodel", False,
    "Whether to export saved model or not")

flags.DEFINE_string(
    "export_savedmodel_dir", "./saved_model",
    "The output directory where the saved model will be exported")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 379,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 379,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, 
                 question_text, 
                 answer_text, 
                 context_text):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          email_text: string. The untokenized text of the email text.
          response_text: string. The untokenized text of the response text.
          label: str. The label ['0','1'] specifying
          if Response belongs to Email or not
        """
        self.guid = guid
        self.question_text = question_text
        self.answer_text = answer_text
        self.context_text = context_text


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it
    means the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 label_mask,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a csv file using pandas."""
        # Read CSV containing E-R pair
        df = pd.read_csv(input_file)
        # get the relevant cols and return dataframe
        df = df[['question',
                 'answer',
                 'context']]

        return df


class QGProcessor(DataProcessor):
    """Processor for the Disney dataset"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(df)):
            guid = "%s-%s" % (set_type, i)
            # Question
            question_text = tokenization.convert_to_unicode(df.question[i])

            # ANSWER
            answer_text = tokenization.convert_to_unicode(
                 df.answer[i])

            # CONTEXT
            context_text = tokenization.convert_to_unicode(df.context[i])
            
            examples.append(InputExample(guid=guid,
                                         question_text=question_text,
                                         answer_text=answer_text,
                                         context_text=context_text))
        return examples

    def _create_test_results(self, data_dir, output_predict_file,
                             question_predictions):
        """Creates test_results csv from model predictions"""
        df = self._read_csv(os.path.join(data_dir, "test.csv"))
        df['generated'] = question_predictions

        df.to_csv(output_predict_file, index=False)


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * (max_seq_length - 1),
            label_mask=[0] * (max_seq_length - 1),
            is_real_example=False)
    
    # Tokenize Context
    tokens_context = tokenizer.tokenize(example.context_text)
    
    # Tokenize Answer
    tokens_answer = tokenizer.tokenize(example.answer_text)

    def _truncate_seq_pair(tokens_context, tokens_answer, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate
        # the longer sequence one token at a time.
        # This makes more sense than truncating an
        # equal percent of tokens from each, since
        # if one sequence is very
        # short then each token that's truncated
        # likely contains more information
        # than a longer sequence.
        while True:
            total_length = len(tokens_context) + len(tokens_answer)
            if total_length <= max_length:
                break
            if len(tokens_context) > len(tokens_answer):
                tokens_context.pop()
            else:
                tokens_answer.pop()

    _truncate_seq_pair(tokens_context, tokens_answer, max_seq_length - 3)

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
    for token in tokens_context:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_answer:
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

    # Tokenize Question
    tokens_question = tokenizer.tokenize(example.question_text)

    # Account for [SEP]  and ignore [CLS]
    if len(tokens_question) > (max_seq_length - 1) - 1:
        tokens_question = tokens_question[0:((max_seq_length-1) - 1)]
    
    # Add [SEP] token
    tokens_question.append("[SEP]")

    label_ids = tokenizer.convert_tokens_to_ids(tokens_question)
    
    # The mask has 1 for real tokens and 0 for padding tokens.
    label_mask = [1] * len(label_ids)

    # Zero-pad up to the sequence length.
    while len(label_ids) < (max_seq_length - 1):
        label_ids.append(0)
        label_mask.append(0)

    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.compat.v1.logging.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))
        tf.compat.v1.logging.info("label_mask: %s" %
                        " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_mask=label_mask,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["label_mask"] = create_int_feature(feature.label_mask)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length - 1], tf.int64),
        "label_mask": tf.io.FixedLenFeature([seq_length - 1], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        # tf.Example only supports tf.int64,
        # but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
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
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 label_ids, label_mask, use_one_hot_embeddings, vocab_size):
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
    # If you want to use the token-level output,
    # use model.get_sequence_output() instead.
    output_layer = model.get_sequence_output()

    # Add a dense layer sequencially
    output_layer = tf.compat.v1.layers.dense(
            output_layer,
            vocab_size,
            activation=None,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    with tf.compat.v1.variable_scope("loss"):

        # ignore the [CLS] token representation
        logits = output_layer[:, 1:, :]

        # log_probs is [B, seq_len-1, vocab_len]
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # one_hot_labels is [B , seq_len-1, vocab_len]
        one_hot_labels = tf.one_hot(
            label_ids, depth=vocab_size, dtype=tf.float32)
        
        # per_example_loss is [B, ]
        label_mask = tf.cast(label_mask, tf.float32)
        per_example_loss = - \
            tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=-1)
        per_example_loss = tf.reduce_sum(
            input_tensor=label_mask * per_example_loss, axis=-1)
        
        # average loss across batch
        loss = tf.reduce_mean(input_tensor=per_example_loss)

        print(loss)
        
        return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, vocab_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_mask = features["label_mask"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(
                features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(input=label_ids), dtype=tf.float32)

        is_training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, label_mask, use_one_hot_embeddings, vocab_size)

        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #   init_string = ""
        #   if var.name in initialized_variable_names:
        #     init_string = ", *INIT_FROM_CKPT*"
        #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                   init_string)

        output_spec = None
        if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps,
                                                     use_tpu)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids,
                          log_probs, is_real_example):
                predictions = tf.argmax(input=log_probs, axis=-1,
                                        output_type=tf.int32)
                predictions = predictions * label_mask
                
                accuracy = tf.compat.v1.metrics.accuracy(
                    labels=label_ids,
                    predictions=predictions,
                    weights=is_real_example)
                loss = tf.compat.v1.metrics.mean(
                    values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss,
                                        label_ids, log_probs,
                                        is_real_example])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
          
        else:
            # Return the predictions and the specification
            # for serving a SavedModel

            predictions = {
                'pred_ids': tf.argmax(input=log_probs, axis=-1,
                                      output_type=tf.int32),
                'probabilities': log_probs,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'predict': tf.compat.v1.estimator.export.PredictOutput(
                        predictions)},
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def build_serving_input_receiver_fn(seq_length, batch_size=None):
    """Returns a serving_input_receiver_fn that can be used during serving."""
    """An input receiver that expects a serialized tf.Example.
  Args:
    seq_length: A fixed size of sequence for each input example.
    batch_size: number of input tensors that will be passed for prediction
  Returns:
    A function that itself returns a ServingInputReceiver with serialized
    input feature
  """
    feature_spec = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length-1], tf.int64),
        "label_mask": tf.io.FixedLenFeature([seq_length-1], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64)
    }

    def serving_input_receiver_fn():
        serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string,
                                               shape=[batch_size],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.io.parse_example(serialized=serialized_tf_example, features=feature_spec)
        return tf.compat.v1.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

    return serving_input_receiver_fn


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, \
            `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)

    processor = QGProcessor()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) /
            FLAGS.train_batch_size *
            FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    vocab_size = tokenizer.get_vocab_size()

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        vocab_size=vocab_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples,
            FLAGS.max_seq_length, 
            tokenizer,
            train_file)
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the
            # number of examples must be a multiple of the batch size, or else
            # examples will get dropped. So we pad with fake examples which are
            # ignored later on. These do NOT count towards the metric
            # (all tf.metrics support a per-instance weight, and
            # these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, FLAGS.max_seq_length,
            tokenizer, eval_file)

        tf.compat.v1.logging.info("***** Running evaluation *****")
        tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        # get the evalution set results
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # used for running inference
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore
            # the number of examples must be a multiple of the batch size,
            # or else examples will get dropped. So we pad with fake examples
            # which are ignored later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples,
                                                FLAGS.max_seq_length,
                                                tokenizer, predict_file)

        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(
            FLAGS.output_dir, "test_results.csv")
        tf.compat.v1.logging.info("***** Predict results *****")
        # store the question predictions
        question_predictions = []

        for (i, prediction) in enumerate(result):
            # get the predicted ids
            pred_ids = prediction["pred_ids"]

            if i >= num_actual_predict_examples:
                break

            pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
            output = []
            for tok in pred_tokens:
                if tok == "[SEP]":
                    break
                output.append(tok)

            question_predictions.append(" ".join(output))

        assert len(question_predictions) == num_actual_predict_examples

        processor._create_test_results(FLAGS.data_dir, output_predict_file,
                                       question_predictions)

    if FLAGS.do_export_savedmodel:
        # used for exporting saved model for TF-Serving
        serving_input_fn = build_serving_input_receiver_fn(
            FLAGS.max_seq_length)
        # Export saved model, and save the vocab file as an extra asset.
        # Since the model itself does not use the vocab file,
        # this file is saved as an extra asset rather than a core asset.
        estimator._export_to_tpu = FLAGS.use_tpu
        estimator.export_savedmodel(
            FLAGS.export_savedmodel_dir, serving_input_fn,
            assets_extra={"vocab.txt": FLAGS.vocab_file},
            strip_default_attrs=True)
        print("\n*****Successfully Exported Saved model*****\n")

       
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
