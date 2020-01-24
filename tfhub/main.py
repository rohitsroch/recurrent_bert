import tensorflow_hub as hub
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer
from tensorflow.keras.models import Model
import math
import numpy as np
import pandas as pd

max_seq_length = 512  # Your choice here
path_to_ckpt = './bert_qg.model'
file_path = './randomQAdata.csv'
bert_model_tfhub = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

qc = "Osama was killed in 2011 by Obama."
q1 = "When was Osama killed?"
q2 = "Who killed Osama?"
ans1 = "Osama was killed in 2011."
ans2 = "Obama was killed by Osama."

def get_vocab_file(bert_model=bert_model_tfhub):
	bert_layer = hub.KerasLayer(bert_model)
	vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
	do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
	return vocab_file, do_lower_case

def tokenize_a_sample(context, ans, ques, format="qg"):
	vocabfile, dolowercase = get_vocab_file()
	tokenizer = FullTokenizer(vocabfile, dolowercase)
	contokens = tokenizer.tokenize(context)
	anstokens = tokenizer.tokenize(ans)
	questokens = tokenizer.tokenize(ques)
	if format == "sqg":
		iptokens = ["[CLS]"] + contokens + ["[SEP]"] + anstokens + ["[SEP]"] + ["[MASK]"]
		optokens = questokens + ["[SEP]"]
	else:
		iptokens = ["[CLS]"] + contokens + ["[SEP]"] + anstokens + ["[SEP]"]
		optokens = questokens + ["[SEP]"]
	return iptokens, optokens

def tokenize_a_pair(context, ans, format="qg"):
	vocabfile, dolowercase = get_vocab_file()
	tokenizer = FullTokenizer(vocabfile, dolowercase)
	contokens = tokenizer.tokenize(context)
	anstokens = tokenizer.tokenize(ans)
	if format == "sqg":
		iptokens = ["[CLS]"] + contokens + ["[SEP]"] + anstokens + ["[SEP]"] + ["[MASK]"]
	else:
		iptokens = ["[CLS]"] + contokens + ["[SEP]"] + anstokens + ["[SEP]"]
	return iptokens

def get_masks(tokens, max_seq_length):
	"""Mask for padding"""
	if len(tokens) > max_seq_length:
		raise IndexError("Token length more than max seq length!")
	return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
	"""Segments: 0 for the first sequence, 1 for the second"""
	if len(tokens) > max_seq_length:
		raise IndexError("Token length more than max seq length!")
	segments = []
	current_segment_id = 0
	for token in tokens:
		segments.append(current_segment_id)
		if token == "[SEP]":
			current_segment_id = 1
	return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, max_seq_length):
	"""Token ids from Tokenizer vocab"""
	vocabfile, dolowercase = get_vocab_file()
	tokenizer=FullTokenizer(vocabfile, dolowercase)
	token_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
	return input_ids


def build(max_seq_length, bert_model=bert_model_tfhub, train_bert=False):

	bert_layer = hub.KerasLayer(bert_model, trainable=train_bert)
	vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
	do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
	tokenizer = FullTokenizer(vocab_file, do_lower_case)

	input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,),        dtype=tf.int32,name="input_word_ids")
	input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_mask")
	segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="segment_ids")
	pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
	dense_output = tf.keras.layers.Dense(units=tokenizer.get_vocab_size(), input_shape=(768,))(sequence_output[:,1:,:])
	output = tf.keras.layers.Softmax(axis=-1, input_shape=(tokenizer.get_vocab_size(),))(dense_output)
	#print(output)
	#realoutput = [input_mask] * output
	model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=output)
	model.build(input_shape=(None, max_seq_length))
	model.compile(optimizer=tf.keras.optimizers.Adam(),
					loss=tf.keras.losses.CategoricalCrossentropy(),
					metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
	model.summary()
	return model

def prepare_data(file_path=file_path):
	vocabfile, dolowercase = get_vocab_file()
	tokenizer = FullTokenizer(vocabfile, dolowercase)
	data = pd.read_csv(file_path, header=[0])
	data_x = []
	data_y = []
	data = data.values
	print(data.shape)
	for i in range(data.shape[0]):
		iptokens, optokens = tokenize_a_sample(context=data[i][0], ans=data[i][2], ques=data[i][1])
		ip_ids = get_ids(iptokens, max_seq_length)
		ip_masks = get_masks(iptokens, max_seq_length)
		ip_segs = get_segments(iptokens, max_seq_length)
		op_ids = get_ids(optokens, max_seq_length - 1)
		op_ids = tf.keras.utils.to_categorical(np.array([op_ids]), num_classes=tokenizer.get_vocab_size())
		data_x.append([np.array([ip_ids]), np.array([ip_masks]), np.array([ip_segs])])
		data_y.append(op_ids)
	data_x = np.array(data_x)
	data_y = np.array(data_y)
	print(data_x.shape, data_y.shape)
	print(data_x[0].shape)
	print(data_x[0][0].shape)
	print(data_x[0][0][0].shape)
	return data_x, data_y

def train(iptokens, optokens, model: tf.keras.Model, num_epochs=5, batch_size=1, max_seq_length=max_seq_length):

	vocabfile, dolowercase = get_vocab_file()
	tokenizer = FullTokenizer(vocabfile, dolowercase)
	input_ids = get_ids(iptokens, max_seq_length)
	input_masks = get_masks(iptokens, max_seq_length)
	input_segments = get_segments(iptokens, max_seq_length)

	optokens = get_ids(optokens, max_seq_length - 1)
	optokens = tf.keras.utils.to_categorical(np.array([optokens]), num_classes=tokenizer.get_vocab_size())

	model.fit(x=[np.array([input_ids]), np.array([input_masks]), np.array([input_segments])], y=optokens, batch_size=batch_size, epochs=num_epochs)
	model.save_weights(path_to_ckpt, overwrite=True)

def inference(iptokens, model: tf.keras.Model, max_seq_length=max_seq_length):
	#model, tokenizer = build(max_seq_length)
	vocabfile, dolowercase = get_vocab_file()
	tokenizer = FullTokenizer(vocabfile, dolowercase)
	model.load_weights(path_to_ckpt)
	input_ids = get_ids(iptokens, max_seq_length)
	input_masks = get_masks(iptokens, max_seq_length)
	input_segments = get_segments(iptokens, max_seq_length)
	output = model.predict([np.array([input_ids]), np.array([input_masks]), np.array([input_segments])], batch_size=1)
	#print("OP shape before argmax:", output.shape)
	output = output.argmax(axis=-1)
	#print(output[0][5], tokenizer.convert_ids_to_tokens([output[0][4]]))
	#print("OP shape after argmax:", output.shape)
	dec_ques = []
	for i in range(max_seq_length - 1):
		if tokenizer.convert_ids_to_tokens([output[0][i]]) == ["[SEP]"]:
			break
		else:
			dec_ques = dec_ques + tokenizer.convert_ids_to_tokens([output[0][i]])
	return dec_ques

iptokens1, optokens1 = tokenize_a_sample(context=qc, ans=ans1, ques=q1)
iptokens2 = tokenize_a_pair(context=qc, ans=ans2)
# prepare_data()
# model = build(max_seq_length)
# train(iptokens=iptokens1, optokens=optokens1, model=model, num_epochs=10)
# print(inference(iptokens=iptokens2, model=model))