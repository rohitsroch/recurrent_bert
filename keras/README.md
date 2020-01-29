## Recurrent BERT

### Please follow this research paper: https://www.aclweb.org/anthology/D19-5821.pdf

### FineTuning the BERT model 

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) **Fine-tuning a BERT model allows you to carry out transfer learning on your dataset for QG task** 

- Open **bert/finetune_bertQG.py** file and update important paths & hyperparameters

```python
 flags.DEFINE_bool("do_train", True, "Whether to run training.") #if you want to train on train.csv
 flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.") #if you want to eval on dev.csv
 flags.DEFINE_bool("do_predict", True,"Whether to run the model in inference mode on the test set.")
 flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")
 flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
 flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")
 flags.DEFINE_float("num_train_epochs", 5.0,"Total number of training epochs to perform.")
 flags.DEFINE_integer("save_checkpoints_steps", 1000,"How often to save the model checkpoint.")
 flags.DEFINE_integer("iterations_per_loop", 1000,"How many steps to make in each estimator call.")
```
- Now download & unzip the BERT-Base cased checkpoint for fine tuning as a initial checkpoints

```console
 $ cd recurrent_bert/
 $ wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
 $ unzip cased_L-12_H-768_A-12.zip
```
- export the paths required for finetuning

```console
 #Let say, I am at /home/rohit_sroch/recurrent_bert/ path (containing email-classification repository folder)
 
 $ export BERT_BASE_DIR=/home/rohit_sroch/recurrent_bert/cased_L-12_H-768_A-12
 $ export DATA_DIR=/home/rohit_sroch/recurrent_bert/data/csv 
 $ mkdir bert/output_logs
 $ export OUTPUT_DIR=/home/rohit_sroch/recurrent_bert/bert/output_logs
 $ mkdir bert/saved_model
 $ export SAVED_MODEL_DIR=/home/rohit_sroch/recurrent_bert/bert/saved_model
```

- Start the training 

```console
 #run the script
 $ python bert/finetune_bertQG.py --data_dir=$DATA_DIR --task_name=$TASK_NAME --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --output_dir=$OUTPUT_DIR --export_savedmodel_dir=$SAVED_MODEL_DIR 
```
