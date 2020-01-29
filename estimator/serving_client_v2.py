# coding=utf-8
# Copyright 2019 Quantiphi R&D Team

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc
import argparse
import os
import collections
import tensorflow as tf
import numpy as np
import pandas as pd
import tokenization
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

""" TF Serving GRPC client script for Scoring model i.e P(y=0,1|E-R) :
   
    Usage:
      python serving_client.py --csv_path <csv_filepath> \
      --task_name <task_name> --vocab_file <vocab_filepath>
    Returns:
       output will be E-R embedding and corresponding score whether R is a
       response of E or not.
      
    Please update the server argument below for grpc call
    Note: Also, follow the blog to understand gRPC & REST API supported by
          Tensorflow ModelServer https://www.tensorflow.org/serving/api_rest 
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, 
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
        self.answer_text = answer_text
        self.context_text = context_text


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad
    because it means the entire output data won't be generated.

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


def create_example(answer_text, context_text, set_type='inference'):
    """Creates example for the inference"""
    guid = "%s-%s" % (set_type, 0)

    # ANSWER
    answer_text = tokenization.convert_to_unicode(answer_text)

    # CONTEXT
    context_text = tokenization.convert_to_unicode(context_text)

    return InputExample(guid=guid,
                        answer_text=answer_text,
                        context_text=context_text)


def convert_single_example(example, max_seq_length,
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

    
    label_ids = [0] * (max_seq_length - 1)
    label_mask = [0] * (max_seq_length - 1)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_mask=label_mask,
        is_real_example=True)
    return feature


def convert_example_to_feature(example,
                               max_seq_length,
                               tokenizer):
    """Convert a `InputExample`s to a Feature"""
    feature = convert_single_example(example,
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

    return tf_example.SerializeToString()


def make_grpc_api_call(context_answer, batch_size, server_host, server_port,
                       vocab_file, max_seq_length=512,
                       do_lower_case=False, server_name="bert_QG",
                       timeout=60.0):

    input_examples = []
    for tup in context_answer:
        context, answer = tup
        input_example = create_example(context, answer)
        input_examples.append(input_example)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)

    serialized_tf_examples = []
    for input_example in input_examples:
        serialized_tf_example = convert_example_to_feature(input_example,
                                                           max_seq_length,
                                                           tokenizer)
        serialized_tf_examples.append(serialized_tf_example)

    if batch_size > len(serialized_tf_examples):
        batch_size = len(serialized_tf_examples)

    num_batches = int(len(serialized_tf_examples) / batch_size)
    if num_batches % batch_size != 0:
        num_batches = num_batches + 1

    # all examples
    pred_questions = []
   
    for i in range(num_batches):
        # get the batch of serialized tf example
        inp_batch = serialized_tf_examples[i*batch_size: (i+1)*batch_size]
        if len(inp_batch) == 0:
            continue

        # update the server host:port for making grpc api call
        # create the RPC stub
        server_grpc = server_host + ':' + server_port
        channel = grpc.insecure_channel(server_grpc)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # create the request object and set the name and signature_name params
        request = predict_pb2.PredictRequest()
        request.model_spec.name = server_name
        request.model_spec.signature_name = 'serving_default'

        # fill in the request object with the necessary data
        request.inputs['examples'].CopyFrom(
            tf.make_tensor_proto(inp_batch, dtype=tf.string,
                              shape=[len(inp_batch)]))

        # sync requests with 30 sec wait before termination
        # result_future = stub.Predict(request, 30.)
        # For async requests
        result_future = stub.Predict.future(request, timeout)
        result_future = result_future.result()

        # Get shape of batch and categories in prediction
        NUM_PREDICTIONS = \
            result_future.outputs['pred_ids'].tensor_shape.dim[0].size
        NUM_TOKENS = \
            result_future.outputs['pred_ids'].tensor_shape.dim[1].size
            
        
        batch_preds = np.reshape(
                    result_future.outputs['pred_ids'].int32_val,
                    (int(NUM_PREDICTIONS), int(NUM_TOKENS)))
        
        for (i, pred_ids) in enumerate(batch_preds):
            
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
            output = []
            for tok in pred_tokens:
                if tok == "[SEP]":
                    break
                output.append(tok)
            question_text = " ".join(output)

            pred_questions.append(question_text)

    return pred_questions


if __name__ == '__main__':
    print('Processing....')

    parser = argparse.ArgumentParser(description='Request Client for Scoring Model \
                                                  using Tensorflow Serving \
                                                  REST API')

    parser.add_argument('--in_csv',
                        type=str,
                        help='Input csv containing email and response cols',
                        required=True)

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size for inference')

    parser.add_argument('--vocab_file',
                        type=str,
                        help='The vocabulary file that the BERT model \
                              was trained on.',
                        required=True)

    parser.add_argument('--server_host',
                        type=str,
                        default='35.231.57.184',
                        help='prediction service host for grpc call')

    parser.add_argument('--server_port',
                        type=str,
                        default='8500',
                        help='prediction service port for grpc call')

    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help='The maximum total input sequence length after \
                             WordPiece tokenization.')

    parser.add_argument('--do_lower_case',
                        type=bool,
                        default=False,
                        help='Whether to lower case the input text. Should be \
                              True for uncased')

    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    context_answer = [(context, answer)
                      for context, answer in zip(df.context, df.answer)]

    pred_questions = make_grpc_api_call(
                                context_answer,
                                args.batch_size,
                                args.server_host,
                                args.server_port,
                                args.vocab_file,
                                args.max_seq_length,
                                args.do_lower_case)
    df['gen_question'] = pred_questions
    df.to_csv('./result.csv', index=False)