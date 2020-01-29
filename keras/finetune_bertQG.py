# Copyright 2019 Rohit Sroch (rohitsroch@gmail.com) Author. All Rights Reserved.

"""BERT-QG in Tensorflow-2.0 using Keras API
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import tensorflow as tf
import tokenization
import tensorflow_hub as hub
import pandas as pd
import distribution_utils
import optimization
from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .csv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model). \
     i.e To initialize BERT hub module graph with these weights")

flags.DEFINE_string(
    "bert_hub_module_handle", "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1",
    "Handle for the BERT TF-Hub module.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_export_savedmodel", True,
    "Whether to export saved model or not")

flags.DEFINE_string(
    "export_savedmodel_dir", "./saved_model",
    "The output directory where the saved model will be exported")

flags.DEFINE_integer("train_batch_size", 1, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 2,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("num_gpus", 0, "Total Number of GPUS")

flags.DEFINE_string(
    "distribution_strategy", "one_device",
    "Distribution strategy for training with one of \
     ['multi_worker_mirrored', 'one_device', 'mirrored', 'parameter_server ]")


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
                 label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask


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
            self._read_csv(os.path.join(data_dir, "train.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "test")

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
        df = self._read_csv(os.path.join(data_dir, "train.csv"))
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
            label_mask=[0] * (max_seq_length - 1))
    
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
        label_mask=label_mask)
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
       
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, batch_size, seq_length, vocab_size, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([seq_length - 1], tf.int64),
        "label_mask": tf.io.FixedLenFeature([seq_length - 1], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        # tf.Example only supports tf.int64,
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        x = {
          'input_ids': example["input_ids"],
          'input_mask':  example["input_mask"],
          'segment_ids': example["segment_ids"]
        }

        y = example['label_ids']

        return (x, y)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
    
    d = d.map(lambda record: _decode_record(record, name_to_features))

    d = d.batch(batch_size, drop_remainder=is_training)
    d = d.prefetch(batch_size)

    return d

def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)


def create_model(is_training, max_seq_length,
                 vocab_size,
                 bert_hub_module_handle):
    """Creates a classification model."""

    bert_layer = hub.KerasLayer(bert_hub_module_handle,
                                 trainable=True)

    # Using Functional API
    input_ids_tensor = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_ids")
    input_mask_tensor = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
    segment_ids_tensor = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
    
    _, sequence_output = bert_layer([input_ids_tensor, 
                                     input_mask_tensor, 
                                     segment_ids_tensor])

    output_layer = tf.keras.layers.Dense(
                        units=vocab_size,
                        activation=tf.nn.softmax)(sequence_output)
    
    output_layer = output_layer[:, 1:, :]
    
    model = tf.keras.Model(
                inputs={'input_ids': input_ids_tensor,
                        'input_mask':input_mask_tensor,
                        'segment_ids':segment_ids_tensor}, 
                outputs=output_layer, 
                name='bert_QG')
    
    model.summary()

    return model, bert_layer
     
   

def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')

    #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, \
            `do_eval` or `do_predict' must be True.")

    tf.io.gfile.makedirs(FLAGS.output_dir)

    processor = QGProcessor()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    vocab_size = tokenizer.get_vocab_size()

    if not FLAGS.distribution_strategy:
       raise ValueError('Distribution strategy has not been specified.')

    strategy = distribution_utils.get_distribution_strategy(
                        distribution_strategy=FLAGS.distribution_strategy,
                        num_gpus=FLAGS.num_gpus)
    
    if FLAGS.do_train:

        train_examples = processor.get_train_examples(FLAGS.data_dir)
        steps_per_epoch = int(len(train_examples) / FLAGS.train_batch_size)
        num_train_steps = int(
            len(train_examples) /
            FLAGS.train_batch_size *
            FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples,
            FLAGS.max_seq_length, 
            tokenizer,
            train_file)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            batch_size=FLAGS.train_batch_size,
            seq_length=FLAGS.max_seq_length,
            vocab_size=vocab_size,
            is_training=True,
            drop_remainder=True)


        with strategy.scope():

            model, sub_model = create_model(True, 
                                FLAGS.max_seq_length,
                                vocab_size, 
                                FLAGS.bert_hub_module_handle)
            
            if FLAGS.init_checkpoint:
                checkpoint = tf.train.Checkpoint(model=sub_model)
                checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()
            
            # use a custom optimizer
            model.optimizer = optimization.create_optimizer(FLAGS.learning_rate, 
                                        steps_per_epoch * FLAGS.num_train_epochs, 
                                        num_warmup_steps)
            optimizer = model.optimizer

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
           
            model.compile(optimizer=optimizer, 
                          loss=loss_fn , 
                          metrics=[metric_fn()])

            summary_dir = os.path.join(FLAGS.output_dir, 'summaries')
            summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
            checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_weights_only=False)

            custom_callbacks = [summary_callback, checkpoint_callback]

            model.fit(
                x=train_input_fn,
                steps_per_epoch=steps_per_epoch,
                epochs=FLAGS.num_train_epochs,
                callbacks=custom_callbacks)
    

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    app.run(main)
