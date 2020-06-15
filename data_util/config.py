import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
START_ENCODING = '[CLS]'
STOP_ENCODING = '[SEP]'




root_dir = os.path.expanduser("/home/vbee/tiennv/Text-Summarizer-Pytorch")

#train_data_path = os.path.join(root_dir, "data/train.bin")
train_data_path = os.path.join(root_dir, "dataset/chunked/train_*")
eval_data_path = os.path.join(root_dir, "dataset/chunked/val_*")
decode_data_path = os.path.join(root_dir, "dataset/chunked/test_*")
vocab_path = os.path.join(root_dir, "dataset/vocab")
log_root = os.path.join(root_dir, "log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 16
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = False
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 600000
max_iterations_eval = 1000

use_gpu=True

lr_coverage=0.15

fix_bug = False

train_log = "log/train_log_nocoverage.txt"

eval_log = 'log/eval_log_nocoverage.txt'

best_model_log = 'log/name_best_model_nocoverage.txt'