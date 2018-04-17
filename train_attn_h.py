import datetime
import os
import time
import numpy as np
from models_attn_h import InterRNN, InterRNN2, IntraRNN, Embed
from datahandler_attn_h import IIRNNDataHandler
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
GPU_NO = 0

# dataset path
HOME = os.path.expanduser('~')
DATASET_PATH = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

# logging of testing results
DATE_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
TIME_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
RUN_NAME = str(DATE_NOW) + '-' + str(TIME_NOW) + '-testing-hierarchical-attn-rnn-' + dataset
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
    MAX_EPOCHS = 200
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
message += "DATASET: " + dataset + " MODEL: hierarchical-attn-RNN"
message += "\nCONFIG: N_ITEMS=" + str(N_ITEMS) + " BATCH_SIZE=" + str(BATCH_SIZE)
message += "\nINTRA_INTERNAL_SIZE=" + str(INTRA_INTERNAL_SIZE) + " INTER_INTERNAL_SIZE=" + str(INTER_INTERNAL_SIZE)
message += "\nN_LAYERS=" + str(N_LAYERS) + " EMBEDDING_SIZE=" + str(EMBEDDING_SIZE)
message += "\nN_SESSIONS=" + str(N_SESSIONS) + " SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS=" + str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT_RATE=" + str(DROPOUT_RATE) + " LEARNING_RATE=" + str(LEARNING_RATE)
print(message)

# initialize embedding table
embed = Embed(N_ITEMS, EMBEDDING_SIZE)
if use_cuda:
    embed = embed.cuda(GPU_NO)
embed_optimizer = optim.Adam(embed.parameters(), lr=LEARNING_RATE)

# initialize inter RNN, inter_rnn has no learnable weights in the baseline model, so no optimizer is needed (adding an optimizer gives an errror)
inter_rnn = InterRNN(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, gpu_no=GPU_NO)
if use_cuda:
    inter_rnn = inter_rnn.cuda(GPU_NO)

# initialize inter RNN 2
inter_rnn2 = InterRNN2(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, gpu_no=GPU_NO)
if use_cuda:
    inter_rnn2 = inter_rnn2.cuda(GPU_NO)
inter_optimizer2 = optim.Adam(inter_rnn2.parameters(), lr=LEARNING_RATE)

# initialize intra RNN
intra_rnn = IntraRNN(N_ITEMS, INTRA_INTERNAL_SIZE, EMBEDDING_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, gpu_no=GPU_NO)
if use_cuda:
    intra_rnn = intra_rnn.cuda(GPU_NO)
intra_optimizer = optim.Adam(intra_rnn.parameters(), lr=LEARNING_RATE)

def run(input, target, session_lengths, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths):
    if intra_rnn.training:
        inter_optimizer2.zero_grad()
        intra_optimizer.zero_grad()
        embed_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))
    target = Variable(torch.LongTensor(target))
    session_lengths = Variable(torch.LongTensor(session_lengths).view(-1, 1)) # by reshaping the length to this, it can be broadcasted and used for division.
    previous_session_batch = Variable(torch.LongTensor(previous_session_batch))
    previous_session_lengths = Variable(torch.LongTensor(previous_session_lengths))
    prevoius_session_counts = Variable(torch.LongTensor(prevoius_session_counts))
    sess_rep_batch = Variable(torch.FloatTensor(sess_rep_batch))
    sess_rep_lengths = Variable(torch.LongTensor(sess_rep_lengths))


    if use_cuda:
        input = input.cuda(GPU_NO)
        target = target.cuda(GPU_NO)
        session_lengths = session_lengths.cuda(GPU_NO)
        previous_session_batch = previous_session_batch.cuda(GPU_NO)
        previous_session_lengths = previous_session_lengths.cuda(GPU_NO)
        prevoius_session_counts = prevoius_session_counts.cuda(GPU_NO)
        sess_rep_batch = sess_rep_batch.cuda(GPU_NO)
        sess_rep_lengths = sess_rep_lengths.cuda(GPU_NO)

    input_embedding = embed(input)

    previous_session_batch = previous_session_batch.transpose(0, 1)     # max_sess_rep x batch_size x max_sess_length
    previous_session_lengths = previous_session_lengths.transpose(0, 1) # max_sess_rep x batch_size
    previous_session_lengths += 1

    inter_hidden = inter_rnn.init_hidden(previous_session_batch.size(1), use_cuda)
    all_session_representations = Variable(torch.zeros(MAX_SESSION_REPRESENTATIONS, previous_session_batch.size(1), INTER_INTERNAL_SIZE)).cuda(GPU_NO)

    """
    for i in range(previous_session_batch.size(0)):
        current_session_batch = previous_session_batch[i]   # batch_size x max_sess_length
        current_session_lengths = previous_session_lengths[i]
        current_session_batch = embed(current_session_batch)
        inter_hidden = inter_rnn.init_hidden(previous_session_batch.size(1), use_cuda)
        session_representations = inter_rnn(inter_hidden, current_session_batch, current_session_lengths, user_list, i, input_embedding, session_lengths)
        all_session_representations[i] = session_representations
    """

    all_session_representations, mean_x = inter_rnn(user_list, input_embedding, session_lengths)

    #print("START")
    #print(all_session_representations[0])
    #print(sess_rep_batch[0])

    #all_session_representations = all_session_representations.transpose(0, 1)
    inter2_hidden = inter_rnn2.init_hidden(previous_session_batch.size(1), use_cuda)
    inter2_output, inter2_hidden = inter_rnn2(inter2_hidden, all_session_representations, prevoius_session_counts)

    # call forward on intra gru layer with hidden state from inter
    intra_hidden = inter2_hidden

    #loss = 0
    
    """
    for i in range(input.size(1)):
        # get input for this time-step
        ee = Variable(torch.LongTensor([i]).expand(input.size(0), 1, EMBEDDING_SIZE))
        if use_cuda:
            ee = ee.cuda(GPU_NO)
        e = torch.gather(input_embedding, 1, ee)

        # run input through intra_rnn
        out, intra_hidden = intra_rnn(e, intra_hidden)

        # if training, compute losses
        if intra_rnn.training:
            b = Variable(torch.LongTensor([i]).expand(input.size(0), 1))
            if use_cuda:
                b = b.cuda(GPU_NO)
            t = torch.gather(target, 1, b)
            loss += masked_cross_entropy_loss(out.squeeze(), t.squeeze()).mean(0)

        # combine outputs
        if i == 0:
            output = out
        else:
            output = torch.cat((output, out), 1)
    """
    

    output, intra_hidden = intra_rnn(input_embedding, intra_hidden)

    top_k_values, top_k_predictions = torch.topk(output, TOP_K)

    if intra_rnn.training:
        loss = 0
        loss += masked_cross_entropy_loss(output.view(-1, N_ITEMS), target.view(-1)).mean(0)    
        loss.backward()

        embed_optimizer.step()
        inter_optimizer2.step()
        intra_optimizer.step()

        return loss.data[0], top_k_predictions, mean_x.data

    return top_k_predictions, mean_x.data

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat, dim=1)
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
    inter_rnn.reset_session_representations()
    _batch_number = 0
    xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths = datahandler.get_next_train_batch()
    embed.train()
    inter_rnn.train()
    inter_rnn2.train()
    intra_rnn.train()
    while len(xinput) > int(BATCH_SIZE / 2):
        _batch_number += 1
        batch_start_time = time.time()

        batch_loss, top_k_predictions, session_representations = run(xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths)

        datahandler.store_user_session_representations(session_representations, user_list)

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
        
        xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths = datahandler.get_next_test_batch()
    embed.eval()
    inter_rnn.eval()
    inter_rnn2.eval()
    intra_rnn.eval()
    while len(xinput) > int(BATCH_SIZE / 2):
        batch_start_time = time.time()
        _batch_number += 1

        top_k_predictions, session_representations = run(xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths)

        datahandler.store_user_session_representations(session_representations, user_list)

        # Evaluate predictions
        tester.evaluate_batch(top_k_predictions, targetvalues, sl)

        # Print some stats during testing
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, previous_session_batch, previous_session_lengths, prevoius_session_counts, user_list, sess_rep_batch, sess_rep_lengths = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, recall5, recall10, recall20, mrr5, mrr10, mrr20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(recall5))
    print("Recall@10 = " + str(recall10))
    print("Recall@20 = " + str(recall20))
    print("MRR@5 = " + str(mrr5))
    print("MRR@10 = " + str(mrr10))
    print("MRR@20 = " + str(mrr20))
    print(test_stats)
    if epoch == 1:
        datahandler.log_config(message)
    datahandler.log_test_stats(epoch, epoch_loss, test_stats)
    tensorboard.scalar_summary('Recall@5', recall5, epoch)
    tensorboard.scalar_summary('Recall@10', recall10, epoch)
    tensorboard.scalar_summary('Recall@20', recall20, epoch)
    tensorboard.scalar_summary('MRR@5', mrr5, epoch)
    tensorboard.scalar_summary('MRR@10', mrr10, epoch)
    tensorboard.scalar_summary('MRR@20', mrr20, epoch)
    tensorboard.scalar_summary('epoch_loss', epoch_loss, epoch)

    epoch += 1
