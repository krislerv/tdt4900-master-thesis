import datetime
import os
import time
import numpy as np
from models_attn_b import InterRNN, IntraRNN, Embed, OnTheFlySessionRepresentations, SessRepEmbed
from datahandler_attn_b import IIRNNDataHandler
from test_util_h import Tester

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from tensorboard import Logger as TensorBoard

# datasets
reddit = "reddit-2-month"
lastfm = "lastfm-3-months"
dataset = lastfm

# GPU settings
use_cuda = True
GPU_NO = 1

method = "ATTN-G"  # LHS, AVG, ATTN-G, ATTN-L

# dataset path
HOME = os.path.expanduser('~')
DATASET_PATH = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

# logging of testing results
DATE_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
TIME_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
RUN_NAME = str(DATE_NOW) + '-' + str(TIME_NOW) + '-testing-attn-rnn-' + dataset
LOG_FILE = './testlog/' + RUN_NAME + '.txt'
tensorboard = TensorBoard('./logs')

# set seed
seed = 0
torch.manual_seed(seed)

# RNN configuration
if dataset == reddit:
    INTRA_INTERNAL_SIZE = 50
    INTER_INTERNAL_SIZE = INTRA_INTERNAL_SIZE
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.0
    MAX_EPOCHS = 31
elif dataset == lastfm:
    INTRA_INTERNAL_SIZE = 100
    INTER_INTERNAL_SIZE = INTRA_INTERNAL_SIZE
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    MAX_EPOCHS = 50
N_LAYERS     = 1
EMBEDDING_SIZE = INTRA_INTERNAL_SIZE
TOP_K = 20
N_ITEMS      = -1
BATCH_SIZE    = 60
MAX_SESSION_REPRESENTATIONS = 15

# Load training data
datahandler = IIRNNDataHandler(DATASET_PATH, BATCH_SIZE, LOG_FILE, MAX_SESSION_REPRESENTATIONS, INTER_INTERNAL_SIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

message = "------------------------------------------------------------------------\n"
message += dataset + " with average of embeddings\n"
message += "DATASET: " + dataset + " MODEL: attn-RNN"
message += "\nCONFIG: N_ITEMS=" + str(N_ITEMS) + " BATCH_SIZE=" + str(BATCH_SIZE)
message += "\nINTRA_INTERNAL_SIZE=" + str(INTRA_INTERNAL_SIZE) + " INTER_INTERNAL_SIZE=" + str(INTER_INTERNAL_SIZE)
message += "\nN_LAYERS=" + str(N_LAYERS) + " EMBEDDING_SIZE=" + str(EMBEDDING_SIZE)
message += "\nN_SESSIONS=" + str(N_SESSIONS) + " SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS=" + str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT_RATE=" + str(DROPOUT_RATE) + " LEARNING_RATE=" + str(LEARNING_RATE)
print(message)

embed = Embed(N_ITEMS, EMBEDDING_SIZE)
if use_cuda:
    embed = embed.cuda(GPU_NO)
embed_optimizer = optim.Adam(embed.parameters(), lr=LEARNING_RATE)

sess_rep_embed = SessRepEmbed(N_ITEMS, EMBEDDING_SIZE)
if use_cuda:
    sess_rep_embed = sess_rep_embed.cuda(GPU_NO)
sess_rep_embed_optimizer = optim.Adam(sess_rep_embed.parameters(), lr=LEARNING_RATE)

# initialize inter RNN
inter_rnn = InterRNN(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, method, gpu_no=GPU_NO)
if use_cuda:
    inter_rnn = inter_rnn.cuda(GPU_NO)
inter_optimizer = optim.Adam(inter_rnn.parameters(), lr=LEARNING_RATE)

# initialize intra RNN
intra_rnn = IntraRNN(N_ITEMS, INTRA_INTERNAL_SIZE, EMBEDDING_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, gpu_no=GPU_NO)
if use_cuda:
    intra_rnn = intra_rnn.cuda(GPU_NO)
intra_optimizer = optim.Adam(intra_rnn.parameters(), lr=LEARNING_RATE)

on_the_fly_sess_reps = OnTheFlySessionRepresentations(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, method, gpu_no=GPU_NO)
if use_cuda:
    on_the_fly_sess_reps = on_the_fly_sess_reps.cuda(GPU_NO)
on_the_fly_sess_reps_optimizer = optim.Adam(on_the_fly_sess_reps.parameters(), lr=LEARNING_RATE)

def run(input, target, session_lengths, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts):
    if intra_rnn.training:
        inter_optimizer.zero_grad()
        intra_optimizer.zero_grad()
        embed_optimizer.zero_grad()
        sess_rep_embed_optimizer.zero_grad()
        on_the_fly_sess_reps_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))
    target = Variable(torch.LongTensor(target))
    session_lengths = Variable(torch.LongTensor(session_lengths).view(-1, 1)) # by reshaping the length to this, it can be broadcasted and used for division.
    session_reps = Variable(torch.FloatTensor(session_reps))
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length))
    #user_list = Variable(torch.LongTensor((user_list).tolist()))
    previous_session_batch = Variable(torch.LongTensor(previous_session_batch))
    previous_session_lengths = Variable(torch.LongTensor(previous_session_lengths))
    prevoius_session_counts = Variable(torch.LongTensor(prevoius_session_counts))


    if use_cuda:
        input = input.cuda(GPU_NO)
        target = target.cuda(GPU_NO)
        session_lengths = session_lengths.cuda(GPU_NO)
        session_reps = session_reps.cuda(GPU_NO)
        inter_session_seq_length = inter_session_seq_length.cuda(GPU_NO)
        #user_list = user_list.cuda(GPU_NO)
        previous_session_batch = previous_session_batch.cuda(GPU_NO)
        previous_session_lengths = previous_session_lengths.cuda(GPU_NO)
        prevoius_session_counts = prevoius_session_counts.cuda(GPU_NO)

        #print("HEI")
        #print(inter_session_seq_length)
        #print(prevoius_session_counts)

    input_embedding = embed(input)
    input_embedding = F.dropout(input_embedding, DROPOUT_RATE, intra_rnn.training, False)

    # The data in input is the last data in pervious_session_batch (as expected)
    #if not intra_rnn.training:
    #    print("------------------------------------------------------------")
    #    print(previous_session_batch[5])
    #    print(previous_session_lengths[5])
    #    print(input[5])
    #    print(session_lengths[5])

    all_session_representations = Variable(torch.zeros(input.size(0), MAX_SESSION_REPRESENTATIONS, INTER_INTERNAL_SIZE)).cuda(GPU_NO)
    for i in range(input.size(0)):
        user_previous_session_batch = previous_session_batch[i]
        user_previous_session_lengths = previous_session_lengths[i]
        user_prevoius_session_counts = prevoius_session_counts[i]

        user_previous_session_batch_embedding = embed(user_previous_session_batch)
        user_previous_session_batch_embedding = F.dropout(user_previous_session_batch_embedding, DROPOUT_RATE, intra_rnn.training, False)

        hidden = on_the_fly_sess_reps.init_hidden(MAX_SESSION_REPRESENTATIONS, use_cuda=use_cuda)

        all_session_representations[i] = on_the_fly_sess_reps(hidden, user_previous_session_batch_embedding, user_previous_session_lengths, user_prevoius_session_counts, user_list[i])

    #if (all_session_representations == session_reps).float().mean().data[0] != 1.0:
    #    print("something fucked")

    inter_hidden = inter_rnn.init_hidden(session_reps.size(0), use_cuda)
    inter_hidden = inter_rnn(all_session_representations, inter_hidden, prevoius_session_counts, user_list)

    # call forward on intra gru layer with hidden state from inter
    intra_hidden = inter_hidden

    output, intra_hidden, input_embedding_d = intra_rnn(input_embedding, intra_hidden)

    if intra_rnn.training:
        loss = 0
        loss += masked_cross_entropy_loss(output.view(-1, N_ITEMS), target.view(-1)).mean(0)    
        loss.backward()

        inter_optimizer.step()
        intra_optimizer.step()
        embed_optimizer.step()
        sess_rep_embed_optimizer.step()
        on_the_fly_sess_reps_optimizer.step()

    # get average pooling of input for session representations
    sum_x = input_embedding_d.sum(1)
    mean_x = sum_x.div(session_lengths.float())

    top_k_values, top_k_predictions = torch.topk(output, TOP_K)

    # return loss and new session representation
    if intra_rnn.training:
        return loss.data[0], mean_x.data, top_k_predictions
    else:
        return mean_x.data, top_k_predictions

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat)
    logpy = torch.gather(logp, 1, y.view(-1, 1))
    mask = Variable(y.data.float().sign().view(-1, 1))
    logpy = logpy * mask
    return logpy.view(-1)

##
##  TRAINING
##
print("Starting training.")

epoch = 1
log_count = 0

print()

num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
while epoch <= MAX_EPOCHS:
    print("Starting epoch #" + str(epoch))
    epoch_loss = 0

    datahandler.reset_user_batch_data()
    datahandler.reset_user_session_representations()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts = datahandler.get_next_train_batch()
    intra_rnn.train()
    inter_rnn.train()
    embed.train()
    sess_rep_embed.train()
    on_the_fly_sess_reps.train()
    while len(xinput) > int(BATCH_SIZE / 2):
        _batch_number += 1
        batch_start_time = time.time()

        batch_loss, sess_rep, top_k_predictions = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts)
        
        datahandler.store_user_session_representations(sess_rep, user_list)

        epoch_loss += batch_loss
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_training_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            print("\t Batch loss:", "%.3f" % batch_loss, end='')
            eta = (batch_runtime * (num_training_batches - _batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")

        #============ TensorBoard logging ============#
        tensorboard.scalar_summary('batch_loss', batch_loss, log_count)
        log_count += 1
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts = datahandler.get_next_test_batch()
    intra_rnn.eval()
    inter_rnn.eval()
    embed.eval()
    sess_rep_embed.eval()
    on_the_fly_sess_reps.eval()
    while len(xinput) > int(BATCH_SIZE / 2):
        batch_start_time = time.time()
        _batch_number += 1

        sess_rep, batch_predictions = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts)

        datahandler.store_user_session_representations(sess_rep, user_list)

        # Evaluate predictions
        tester.evaluate_batch(batch_predictions, targetvalues, sl)

        # Print some stats during testing
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall10, current_recall20, mrr5, mrr10, mrr20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print(test_stats)
    if epoch == 1:
        datahandler.log_config(message)
    datahandler.log_test_stats(epoch, epoch_loss, test_stats)
    tensorboard.scalar_summary('Recall@5', current_recall5, epoch)
    tensorboard.scalar_summary('Recall@10', current_recall5, epoch)
    tensorboard.scalar_summary('Recall@20', current_recall20, epoch)
    tensorboard.scalar_summary('MRR@5', mrr5, epoch)
    tensorboard.scalar_summary('MRR@10', mrr10, epoch)
    tensorboard.scalar_summary('MRR@20', mrr20, epoch)
    tensorboard.scalar_summary('epoch_loss', epoch_loss, epoch)

    epoch += 1
