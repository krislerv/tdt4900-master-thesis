import datetime
import os
import time
import numpy as np
from models_attn_base import InterRNN, IntraRNN
from datahandler_inter import IIRNNDataHandler
from test_util import Tester

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from tensorboard import Logger as TensorBoard

# datasets
reddit = "subreddit"
lastfm = "lastfm"
dataset = lastfm

# which type of session representation to use. False: Average pooling, True: Last hidden state
use_last_hidden_state = False

# use gpu
use_cuda = True

# dataset path
HOME = os.path.expanduser('~')
DATASET_PATH = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

# logging
DATE_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
LOG_FILE = './testlog/' + str(DATE_NOW) + '-testing-attn-rnn-' + dataset + '.txt'
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
BATCH_SIZE    = 100
MAX_SESSION_REPRESENTATIONS = 15

# Load training data
datahandler = IIRNNDataHandler(DATASET_PATH, BATCH_SIZE, LOG_FILE, MAX_SESSION_REPRESENTATIONS, INTER_INTERNAL_SIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

message = "------------------------------------------------------------------------\n"
if use_last_hidden_state:
    message += dataset + " with last hidden state\n"
else:
    message += dataset + " with average of embeddings\n"
message += "DATASET: " + dataset + " MODEL: attn-II-RNN"
message += "\nCONFIG: N_ITEMS=" + str(N_ITEMS) + " BATCH_SIZE=" + str(BATCH_SIZE)
message += "\nINTRA_INTERNAL_SIZE=" + str(INTRA_INTERNAL_SIZE) + " INTER_INTERNAL_SIZE=" + str(INTER_INTERNAL_SIZE)
message += "\nN_LAYERS=" + str(N_LAYERS) + " EMBEDDING_SIZE=" + str(EMBEDDING_SIZE)
message += "\nN_SESSIONS=" + str(N_SESSIONS) + " SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS=" + str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT_RATE=" + str(DROPOUT_RATE) + " LEARNING_RATE=" + str(LEARNING_RATE)
print(message)

# initialize inter RNN
inter_rnn = InterRNN(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE)
if use_cuda:
    inter_rnn = inter_rnn.cuda()
inter_optimizer = optim.Adam(inter_rnn.parameters(), lr=LEARNING_RATE)

# initialize intra RNN
intra_rnn = IntraRNN(N_ITEMS, INTRA_INTERNAL_SIZE, EMBEDDING_SIZE, N_LAYERS, DROPOUT_RATE)
if use_cuda:
    intra_rnn = intra_rnn.cuda()
intra_optimizer = optim.Adam(intra_rnn.parameters(), lr=LEARNING_RATE)

def train(input, target, session_lengths, session_reps, inter_session_seq_length, use_last_hidden_state):
    inter_optimizer.zero_grad()
    intra_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))
    target = Variable(torch.LongTensor(target))
    session_lengths = Variable(torch.LongTensor(session_lengths).view(-1, 1)) # by reshaping the length to this, it can be broadcasted and used for division.
    session_reps = Variable(torch.FloatTensor(session_reps))
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length))

    if use_cuda:
        input = input.cuda()
        target = target.cuda()
        session_lengths = session_lengths.cuda()
        session_reps = session_reps.cuda()
        inter_session_seq_length = inter_session_seq_length.cuda()

    inter_hidden = inter_rnn.init_hidden(session_reps.size(0), use_cuda)
    inter_output, inter_hidden = inter_rnn(session_reps, inter_hidden, inter_session_seq_length)

    loss = 0

    # call forward on intra gru layer with hidden state from inter
    intra_hidden = inter_hidden
    for i in range(input.size(1)):
        b = torch.LongTensor([i]).expand(input.size(0), 1)
        if use_cuda:
            b = b.cuda()
        c = torch.gather(input, 1, b)
        t = torch.gather(target, 1, b)
        out, intra_hidden, embedded_input, gru = intra_rnn(c, intra_hidden, inter_output)
        loss += masked_cross_entropy_loss(out.squeeze(), t.squeeze()).mean(0)
        if i == 0:
            gru_output = gru
            cat_embedded_input = embedded_input
        else:
            gru_output = torch.cat((gru_output, gru), 1)
            cat_embedded_input = torch.cat((cat_embedded_input, embedded_input), 1)

    # get last hidden states for session representations
    last_index_of_sessions = session_lengths - 1
    hidden_indices = last_index_of_sessions.view(-1, 1, 1).expand(gru_output.size(0), 1, gru_output.size(2))
    hidden_out = torch.gather(gru_output, 1, hidden_indices)
    hidden_out = hidden_out.squeeze()
    hidden_out = hidden_out.unsqueeze(0)

    # get average pooling of input for session representations
    sum_x = cat_embedded_input.sum(1)
    mean_x = sum_x.div(session_lengths.float())

    loss.backward()

    inter_optimizer.step()
    intra_optimizer.step()

    # return loss and new session representation
    if use_last_hidden_state:
        return loss.data[0], hidden_out.data[0]
    return loss.data[0], mean_x.data

def predict(input, session_lengths, session_reps, inter_session_seq_length):
    input = Variable(torch.LongTensor(input))
    session_lengths = Variable(torch.LongTensor(session_lengths).view(-1, 1)) # by reshaping the length to this, it can be broadcasted and used for division.
    session_reps = Variable(torch.FloatTensor(session_reps))
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length))

    if use_cuda:
        input = input.cuda()
        session_lengths = session_lengths.cuda()
        session_reps = session_reps.cuda()
        inter_session_seq_length = inter_session_seq_length.cuda()

    inter_hidden = inter_rnn.init_hidden(session_reps.size(0), use_cuda)
    inter_output, inter_hidden = inter_rnn(session_reps, inter_hidden, inter_session_seq_length)

    intra_hidden = inter_hidden
    for i in range(input.size(1)):
        b = torch.LongTensor([i]).expand(input.size(0), 1)
        if use_cuda:
            b = b.cuda()
        c = torch.gather(input, 1, b)
        out, intra_hidden, embedded_input, gru = intra_rnn(c, intra_hidden, inter_output)
        if i == 0:
            output = out
            gru_output = gru
            cat_embedded_input = embedded_input
        else:
            output = torch.cat((output, out), 1)
            gru_output = torch.cat((gru_output, gru), 1)
            cat_embedded_input = torch.cat((cat_embedded_input, embedded_input), 1)

    # get last hidden states for session representations
    last_index_of_sessions = session_lengths - 1
    hidden_indices = last_index_of_sessions.view(-1, 1, 1).expand(gru_output.size(0), 1, gru_output.size(2))
    hidden_out = torch.gather(gru_output, 1, hidden_indices)
    hidden_out = hidden_out.squeeze()
    hidden_out = hidden_out.unsqueeze(0)

    # get average pooling of input for session representations
    sum_x = cat_embedded_input.sum(1)
    mean_x = sum_x.div(session_lengths.float())

    top_k_values, top_k_predictions = torch.topk(output, TOP_K)

    if use_last_hidden_state:
        return top_k_predictions, hidden_out.data[0]
    return top_k_predictions, mean_x.data

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat)
    logpy = torch.gather(logp, 1, y.view(-1, 1))
    mask = Variable(y.data.float().sign().view(-1, 1))
    logpy = logpy * mask
    return logpy.view(-1)

def to_np(x):
    return x.data.cpu().numpy()

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
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list = datahandler.get_next_train_batch()
    intra_rnn.train()
    inter_rnn.train()
    while len(xinput) > int(BATCH_SIZE / 2):
        _batch_number += 1
        batch_start_time = time.time()
        batch_loss, sess_rep = train(xinput, targetvalues, sl, session_reps, inter_session_seq_length, use_last_hidden_state)

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
            """
            for tag, value in inter_rnn.named_parameters():
                tag = tag.replace('.', '/')
                tensorboard.histo_summary('inter/' + tag, to_np(value), log_count)
                tensorboard.histo_summary('inter/' + tag + '/grad', to_np(value.grad), log_count)
            for tag, value in intra_rnn.named_parameters():
                tag = tag.replace('.', '/')
                tensorboard.histo_summary('intra/' + tag, to_np(value), log_count)
                tensorboard.histo_summary('intra/' + tag + '/grad', to_np(value.grad), log_count)
            """
            log_count += 1
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list = datahandler.get_next_test_batch()
    intra_rnn.eval()
    inter_rnn.eval()
    while len(xinput) > int(BATCH_SIZE / 2):
        batch_start_time = time.time()
        _batch_number += 1

        batch_predictions, sess_rep = predict(xinput, sl, session_reps, inter_session_seq_length)

        datahandler.store_user_session_representations(sess_rep, user_list)

        # Evaluate predictions
        prediction_results = tester.evaluate_batch(batch_predictions, targetvalues, sl)
        #print(prediction_results)

        # Print some stats during testing
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    if epoch == 1:
        datahandler.log_config(message)
    datahandler.log_test_stats(epoch, epoch_loss, test_stats)
    tensorboard.scalar_summary('recall@5', current_recall5, epoch)
    tensorboard.scalar_summary('recall@20', current_recall20, epoch)
    tensorboard.scalar_summary('epoch_loss', epoch_loss, epoch)

    epoch += 1
