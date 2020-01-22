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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
from enum import Enum
from absl import app
from absl import flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_export_savedmodel", False,
    "Whether to export saved model or not")

flags.DEFINE_string(
    "export_savedmodel_dir", "./saved_model",
    "The output directory where the saved model will be exported")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 77,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 2e-4,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

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


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        if is_predict is False:
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_weights = features["masked_lm_weights"]
            next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
             bert_config, model.get_sequence_output(),
             model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

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

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps,
                num_warmup_steps, use_tpu)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights,
                          next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                                    masked_lm_log_probs,
                                    [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(
                    masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs,
                    [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    input=next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=next_sentence_labels,
                    predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.compat.v1.metrics.mean(
                    values=next_sentence_example_loss)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            ])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
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
                elif FLAGS.pooling_strategy == \
                        PoolingStrategy.REDUCE_MAX.value:
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                    predictions["embedding"] = pooled
                elif FLAGS.pooling_strategy == \
                        PoolingStrategy.REDUCE_MEAN_MAX.value:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer,
                                                           input_mask),
                                        masked_reduce_max(encoder_layer,
                                                          input_mask)], axis=1)
                    predictions["embedding"] = pooled
                elif FLAGS.pooling_strategy == \
                        PoolingStrategy.FIRST_TOKEN.value or \
                        FLAGS.pooling_strategy == \
                        PoolingStrategy.CLS_TOKEN.value:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                    predictions["embedding"] = pooled
                elif FLAGS.pooling_strategy == \
                        PoolingStrategy.LAST_TOKEN.value or \
                        FLAGS.pooling_strategy == \
                        PoolingStrategy.SEP_TOKEN.value:
                    seq_len = tf.cast(tf.reduce_sum(
                        input_tensor=input_mask, axis=1), tf.int32)
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


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.compat.v1.variable_scope("transform"):
            input_tensor = tf.compat.v1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.compat.v1.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = - \
            tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
        denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("cls/seq_relationship"):
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(
                            bert_config.initializer_range))
        output_bias = tf.compat.v1.get_variable(
            "output_bias", shape=[2], initializer=tf.compat.v1.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(input_tensor=per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.io.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't
            # want to encounter out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the
        # CPU or GPU and we *don't* want to drop the remainder, otherwise
        # we wont cover every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    return example


def build_serving_input_receiver_fn(seq_length, batch_size=None):
    """Returns a serving_input_receiver_fn that can be used during serving."""
    """An input receiver that expects a serialized tf.Example.
  Args:
    seq_length: A fixed size of sequence for each input example.
    batch_size: number of input tensors that will be passed for prediction
  Returns:
    A function that itself returns a ServingInputReceiver
    with serialized input feature
  """
    feature_spec = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64)
    }

    def serving_input_receiver_fn():
        serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.string,
                                               shape=[batch_size],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.io.parse_example(serialized=serialized_tf_example, features=feature_spec)
        return tf.estimator.export.ServingInputReceiver(features,
                                                        receiver_tensors)

    return serving_input_receiver_fn


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    tf.compat.v1.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.compat.v1.logging.info("  %s" % input_file)

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

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn,
                        max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.compat.v1.logging.info("***** Running evaluation *****")
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

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
            strip_default_attrs=True)
        tf.compat.v1.logging.info("*****Successfully Exported Saved model*****")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
