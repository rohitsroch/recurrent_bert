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
"""Extract pre-computed feature vectors from BERT.
   NOTE: Takes csv as input containing emails
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import tokenization
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from enum import Enum
from absl import app
from absl import flags

flags.DEFINE_string("input_csv", None, "Input CSV file with email col")

flags.DEFINE_string("output_dir", "./embeddings",
                    "Output dir for saving embedding as pickle file")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 2, "Batch size for predictions.")

# -1 = LAST LAYER #This is by default
# -2 = SECOND TO LAST LAYER
# -3 = THIRD TO LAST LAYER
# -4 = FOURTH TO LAST LAYER
flags.DEFINE_list("pooling_layer", [-1],
                  "the encoder layer(s) that receives pooling \
                   with choices [-1,-2,-3,-4]")

# REDUCE_MEAN = 1  #This is by default
# REDUCE_MAX = 2
# REDUCE_MEAN_MAX = 3
flags.DEFINE_integer("pooling_strategy", 1,
                     "the pooling strategy for generating encoding vectors \
                     with choices 0,1,2,3,4,5")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MEAN = 1
    REDUCE_MAX = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


class InputExample(object):

    def __init__(self, guid, email_text):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          email_text: string. The untokenized text of the email text sequence.
        """
        self.guid = guid
        self.email_text = email_text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_real_example = is_real_example


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it
    means the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_is_real_example = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_is_real_example.append(feature.is_real_example)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets.
        # We do not use Dataset.from_generator() because that uses tf.py_func
        # which is not TPU compatible. The right way to load data is with
        # TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "is_real_example":
                tf.constant(
                    all_is_real_example,
                    shape=[num_examples],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        # is_real_example = features["is_real_example"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.compat.v1.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
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

        predictions = {

        }

        def minus_mask(x, m): return x - \
            tf.expand_dims(1.0 - m, axis=-1) * 1e30

        def mul_mask(x, m): return x * tf.expand_dims(m, axis=-1)

        def masked_reduce_max(x, m): return tf.reduce_max(input_tensor=minus_mask(x, m),
                                                          axis=1)

        def masked_reduce_mean(x, m): return tf.reduce_sum(input_tensor=mul_mask(x, m),
                                                           axis=1) / (
            tf.reduce_sum(input_tensor=m, axis=1, keepdims=True) + 1e-10)

        # encoder_layer = model.all_encoder_layers[FLAGS.pooling_layer[0]]
        # input_mask = tf.cast(input_mask, tf.float32)
        # pooled = masked_reduce_mean(encoder_layer, input_mask)

        with tf.compat.v1.variable_scope("pooling"):
            if len(FLAGS.pooling_layer) == 1:
                encoder_layer = model.all_encoder_layers[
                                FLAGS.pooling_layer[0]]
            else:
                all_layers = [model.all_encoder_layers[l]
                              for l in FLAGS.pooling_layer]
                encoder_layer = tf.concat(all_layers, -1)

            input_mask = tf.cast(input_mask, tf.float32)
            if FLAGS.pooling_strategy == PoolingStrategy.REDUCE_MEAN.value:
                pooled = masked_reduce_mean(encoder_layer, input_mask)
                predictions["embedding"] = pooled
            elif FLAGS.pooling_strategy == PoolingStrategy.REDUCE_MAX.value:
                pooled = masked_reduce_max(encoder_layer, input_mask)
                predictions["embedding"] = pooled
            elif FLAGS.pooling_strategy == \
                    PoolingStrategy.REDUCE_MEAN_MAX.value:
                pooled = tf.concat([masked_reduce_mean(encoder_layer,
                                                       input_mask),
                                    masked_reduce_max(encoder_layer,
                                                      input_mask)], axis=1)
                predictions["embedding"] = pooled
            elif FLAGS.pooling_strategy == PoolingStrategy.FIRST_TOKEN.value \
                    or FLAGS.pooling_strategy == \
                    PoolingStrategy.CLS_TOKEN.value:
                pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                predictions["embedding"] = pooled
            elif FLAGS.pooling_strategy == PoolingStrategy.LAST_TOKEN.value \
                    or FLAGS.pooling_strategy == \
                    PoolingStrategy.SEP_TOKEN.value:
                seq_len = tf.cast(tf.reduce_sum(input_tensor=input_mask, axis=1), tf.int32)
                rng = tf.range(0, tf.shape(input=seq_len)[0])
                indexes = tf.stack([rng, seq_len - 1], 1)
                pooled = tf.gather_nd(encoder_layer, indexes)
                predictions["embedding"] = pooled
            elif FLAGS.pooling_strategy == PoolingStrategy.NONE:
                pooled = mul_mask(encoder_layer, input_mask)
                predictions["embedding"] = pooled

        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)},
            scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            is_real_example=False)

    # ----------------------------------------------
    # tokenize the EMAIL into individual sentence

    # i.e [This is sentence1, This is sentence2,...]
    email_sentences = bert_util.sentence_tokenizer(example.email_text)
    email_total_sen = len(email_sentences)
    # tokenize each of the sentence into individual tokens
    # i.e [ [This, is, ##tence1], [This, is, ##tence2],...]
    # here email_tokens represents tokens of all sentences
    email_tokens = []
    for index in range(email_total_sen):
        for token in tokenizer.tokenize(email_sentences[index]):
            email_tokens.append(token)
        if index < email_total_sen - 1:
            email_tokens.append("[SEP]")

    # Account for [CLS] and [SEP] with "- 2"
    if len(email_tokens) > max_seq_length - 2:
        if email_tokens[0: (max_seq_length - 2)][-1] == "[SEP]":
            email_tokens = email_tokens[0: (max_seq_length - 3)]
        else:
            email_tokens = email_tokens[0: (max_seq_length - 2)]

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
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    switch_ = True
    for token in email_tokens:
        tokens.append(token)
        if switch_ is True:
            segment_ids.append(0)
        else:
            segment_ids.append(1)

        if token == "[SEP]" and switch_ is True:
            switch_ = False
        elif token == "[SEP]" and switch_ is False:
            switch_ = True

    # last token
    tokens.append("[SEP]")
    if switch_ is True:
        segment_ids.append(0)
        switch_ = False
    else:
        segment_ids.append(1)
        switch_ = True

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        if switch_ is True:
            segment_ids.append(0)
        else:
            segment_ids.append(1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

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

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_real_example=True)

    return feature


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to InputFeatures"""
    features = []
    for (ex_index, example) in enumerate(examples):

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)
        features.append(feature)

    return features


def create_examples(in_csv, set_type='inference'):
    """Creates examples """
    examples = []
    df = pd.read_csv(in_csv)
    df = df.sample(frac=1.0).reset_index(drop=True)
    for i in range(len(df)):
        guid = "%s-%s" % (set_type, i)
        # EMAIL
        email_text = tokenization.convert_to_unicode(df.email[i])
        examples.append(InputExample(guid=guid,
                                     email_text=email_text))

    return examples


# show the semantic textual similarity in a heat map.
def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    print(corr)
    sns.set(font_scale=0.8)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    plt.show()


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    examples = create_examples(FLAGS.input_csv)

    features = convert_examples_to_features(
        examples=examples, max_seq_length=FLAGS.max_seq_length,
        tokenizer=tokenizer)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(
        features=features, seq_length=FLAGS.max_seq_length)

    result = estimator.predict(input_fn=input_fn)
    tf.io.gfile.makedirs(FLAGS.output_dir)
    tf.compat.v1.logging.info("***** Predict results *****")

    # BERT Base embeding is 768 dimension
    embedding_arr = np.zeros((len(examples), 768), dtype=np.float32)

    for (i, prediction) in enumerate(result):
        embedding = prediction["embedding"]
        # normalized embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding_arr[i] = embedding
        # np.save(os.path.join(FLAGS.output_dir, "email_{}".format(i)),
        #         embedding)

    # visualize semantic textual similarity
    labels = [example.email_text for example in examples]
    # plot_similarity(labels, embedding_arr, 90)

    output_dict = {}
    count = 0
    for index, val in zip(labels, embedding_arr):
        output_dict[count] = {'email': index, 'embedding': val}
        count += 1

    output_path = os.path.join(FLAGS.output_dir, 'trained_emb.pickle')
    with open(output_path, 'wb') as writer:
        pickle.dump(output_dict, writer)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_csv")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
