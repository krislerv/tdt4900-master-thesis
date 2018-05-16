import datetime
import os
import time
import numpy as np
from models_attn import InterRNN, IntraRNN, Embed
from datahandler_attn import IIRNNDataHandler
from test_util_h import Tester

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from tensorboard import Logger as TensorBoard

# datasets
reddit = "subreddit"
lastfm = "lastfm-full"
dataset = reddit

# which type of session representation to use. False: Average pooling, True: Last hidden state
if dataset == lastfm:
    use_last_hidden_state = False
else:
    use_last_hidden_state = True

bidirectional = True

# Inter-session attention mechanisms
use_hidden_state_attn = False
use_delta_t_attn = True
use_week_time_attn = False

# Intra-session attention mechanisms
use_intra_attn = False
intra_attn_method = "cat"   # options: cat, sum
use_per_user_intra_attn = False # not used if use_intra_attn is False

# logging of attention weights
log_inter_attn = False
log_intra_attn = False

# saving/loading of model parameters
save_model_parameters = True
resume_model = True
resume_model_name = "2018-05-12-10-21-24-testing-attn-rnn-subreddit-False-False"    # unused if resume_model is False

# GPU settings
use_cuda = True
GPU_NO = 0

# dataset path
HOME = os.path.expanduser('~')
DATASET_PATH = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

# logging of testing results
DATE_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
TIME_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
if resume_model:
    RUN_NAME = resume_model_name
else:   # pre 2018-03-06: Three boolean values were inter attn mechanisms, post: they are intra attn mechanisms
    RUN_NAME = str(DATE_NOW) + '-' + str(TIME_NOW) + '-testing-attn-rnn-' + dataset + '-' + str(use_intra_attn) + '-' + str(use_per_user_intra_attn)
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
message += "DATASET: " + dataset + " MODEL: attn-RNN"
message += "\nuse_hidden_state_attn=" + str(use_hidden_state_attn) + " use_delta_t_attn=" + str(use_delta_t_attn) + " use_week_time_attn=" + str(use_week_time_attn)
message += "\nuse_intra_attn=" + str(use_intra_attn) + " intra_attn_method=" + intra_attn_method + " use_per_user_intra_attn=" + str(use_per_user_intra_attn)
message += "\nCONFIG: N_ITEMS=" + str(N_ITEMS) + " BATCH_SIZE=" + str(BATCH_SIZE)
message += "\nINTRA_INTERNAL_SIZE=" + str(INTRA_INTERNAL_SIZE) + " INTER_INTERNAL_SIZE=" + str(INTER_INTERNAL_SIZE)
message += "\nN_LAYERS=" + str(N_LAYERS) + " EMBEDDING_SIZE=" + str(EMBEDDING_SIZE)
message += "\nN_SESSIONS=" + str(N_SESSIONS) + " SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS=" + str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT_RATE=" + str(DROPOUT_RATE) + " LEARNING_RATE=" + str(LEARNING_RATE)
message += "\nbidirectional=" + str(bidirectional)
print(message)

embed = Embed(N_ITEMS, EMBEDDING_SIZE)
if use_cuda:
    embed = embed.cuda(GPU_NO)
embed_optimizer = optim.Adam(embed.parameters(), lr=LEARNING_RATE)

# initialize inter RNN
inter_rnn = InterRNN(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, bidirectional, use_hidden_state_attn=use_hidden_state_attn, use_delta_t_attn=use_delta_t_attn, use_week_time_attn=use_week_time_attn, gpu_no=GPU_NO)
if use_cuda:
    inter_rnn = inter_rnn.cuda(GPU_NO)
inter_optimizer = optim.Adam(inter_rnn.parameters(), lr=LEARNING_RATE)

# initialize intra RNN
intra_rnn = IntraRNN(N_ITEMS, INTRA_INTERNAL_SIZE, EMBEDDING_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, bidirectional, use_attn=use_intra_attn, use_per_user_intra_attn=use_per_user_intra_attn, intra_attn_method=intra_attn_method, gpu_no=GPU_NO)
if use_cuda:
    intra_rnn = intra_rnn.cuda(GPU_NO)
intra_optimizer = optim.Adam(intra_rnn.parameters(), lr=LEARNING_RATE)

if resume_model:
    embed.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-embed_model.pth"))
    embed_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-embed_optimizer.pth"))
    inter_rnn.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-inter_model.pth"))
    inter_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-inter_optimizer.pth"))
    intra_rnn.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-intra_model.pth"))
    intra_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-intra_optimizer.pth"))

def run(input, target, session_lengths, session_reps, inter_session_seq_length, input_timestamps, input_timestamp_bucket_ids, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list):
    if intra_rnn.training:
        inter_optimizer.zero_grad()
        intra_optimizer.zero_grad()
        embed_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))
    target = Variable(torch.LongTensor(target))
    session_lengths = Variable(torch.LongTensor(session_lengths).view(-1, 1)) # by reshaping the length to this, it can be broadcasted and used for division.
    session_reps = Variable(torch.FloatTensor(session_reps))
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length))
    input_timestamps = Variable(torch.FloatTensor(input_timestamps))
    sess_rep_timestamps_batch = Variable(torch.FloatTensor(sess_rep_timestamps_batch))
    sess_rep_timestamp_bucket_ids_batch = Variable(torch.LongTensor(sess_rep_timestamp_bucket_ids_batch))
    user_list = Variable(torch.LongTensor((user_list).tolist()))

    if use_cuda:
        input = input.cuda(GPU_NO)
        target = target.cuda(GPU_NO)
        session_lengths = session_lengths.cuda(GPU_NO)
        session_reps = session_reps.cuda(GPU_NO)
        inter_session_seq_length = inter_session_seq_length.cuda(GPU_NO)
        input_timestamps = input_timestamps.cuda(GPU_NO)
        sess_rep_timestamps_batch = sess_rep_timestamps_batch.cuda(GPU_NO)
        sess_rep_timestamp_bucket_ids_batch = sess_rep_timestamp_bucket_ids_batch.cuda(GPU_NO)
        user_list = user_list.cuda(GPU_NO)

    input_embedding = embed(input)

    input_timestamps = input_timestamps.unsqueeze(1).expand(input.size(0), MAX_SESSION_REPRESENTATIONS)
    delta_t = input_timestamps - sess_rep_timestamps_batch
    delta_t_hours = delta_t.div(3600)

    delta_t_hours = delta_t_hours.floor().long()
    delta_t_ceiling = Variable(torch.LongTensor([168]).expand(input.size(0), 15)).cuda(GPU_NO)    # 168 hours in a week
    delta_t_hours = torch.min(delta_t_hours, delta_t_ceiling)

    summed_delta_t_hours = torch.sum(delta_t_hours, 1)
    average_delta_t_hours = summed_delta_t_hours.div(MAX_SESSION_REPRESENTATIONS)

    std_delta_t_hours = delta_t_hours.float().std(dim=1)

    inter_hidden = inter_rnn.init_hidden(session_reps.size(0), use_cuda)
    inter_output, inter_hidden, inter_attn_weights = inter_rnn(session_reps, inter_hidden, inter_session_seq_length, delta_t_hours, sess_rep_timestamp_bucket_ids_batch)

    # call forward on intra gru layer with hidden state from inter
    intra_hidden = inter_hidden

    if use_intra_attn:
        if intra_rnn.training:
            loss = 0
        output = Variable(torch.zeros(19, input_embedding.size(0), N_ITEMS)).cuda(GPU_NO)
        gru_output = Variable(torch.zeros(19, input_embedding.size(0), (1 + bidirectional) * INTRA_INTERNAL_SIZE)).cuda(GPU_NO)
        cat_embedded_input = Variable(torch.zeros(19, input_embedding.size(0), INTRA_INTERNAL_SIZE)).cuda(GPU_NO)
        intra_attn_weights = Variable(torch.zeros(19, input_embedding.size(0), MAX_SESSION_REPRESENTATIONS)).cuda(GPU_NO)
        for i in range(input.size(1)):
            b = Variable(torch.LongTensor([i]).expand(input.size(0), 1))
            ee = Variable(torch.LongTensor([i]).expand(input.size(0), 1, EMBEDDING_SIZE))
            if use_cuda:
                b = b.cuda(GPU_NO)
                ee = ee.cuda(GPU_NO)
            e = torch.gather(input_embedding, 1, ee)
            t = torch.gather(target, 1, b)
            out, intra_hidden, embedded_input, gru, attn_weights = intra_rnn(e, intra_hidden, inter_output, delta_t_hours, user_list)
            if intra_rnn.training:
                loss += masked_cross_entropy_loss(out.squeeze(), t.squeeze()).mean(0)
            output[i] = out
            gru_output[i] = gru
            cat_embedded_input[i] = embedded_input
            intra_attn_weights[i] = attn_weights.unsqueeze(1)

        output = output.transpose(0, 1)
        gru_output = gru_output.transpose(0, 1)
        cat_embedded_input = cat_embedded_input.transpose(0, 1)
        intra_attn_weights = intra_attn_weights.transpose(0, 1)

        if intra_rnn.training:
            loss.backward()

            embed_optimizer.step()
            inter_optimizer.step()
            intra_optimizer.step()

    else:
        output, intra_hidden, cat_embedded_input, gru_output, intra_attn_weights = intra_rnn(input_embedding, intra_hidden, inter_output, delta_t_hours, user_list)
        if intra_rnn.training:
            loss = 0
            loss += masked_cross_entropy_loss(output.view(-1, N_ITEMS), target.view(-1)).mean(0)    
            loss.backward()

            inter_optimizer.step()
            intra_optimizer.step()
            embed_optimizer.step()
    
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

    # return loss and new session representation
    if intra_rnn.training:
        if use_last_hidden_state:
            return loss.data[0], hidden_out.data[0], inter_attn_weights, intra_attn_weights, top_k_predictions
        return loss.data[0], mean_x.data, inter_attn_weights, intra_attn_weights, top_k_predictions
    else:
        if use_last_hidden_state:
            return hidden_out.data[0], inter_attn_weights, intra_attn_weights, top_k_predictions
        return mean_x.data, inter_attn_weights, intra_attn_weights, top_k_predictions


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
    xinput, targetvalues, sl, input_timestamps, input_timestamp_bucket_ids, session_reps, inter_session_seq_length, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list = datahandler.get_next_train_batch()
    intra_rnn.train()
    inter_rnn.train()
    embed.train()
    while len(xinput) > int(BATCH_SIZE / 2):
        _batch_number += 1
        batch_start_time = time.time()

        batch_loss, sess_rep, inter_attn_weights, intra_attn_weights, top_k_predictions = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, input_timestamps, input_timestamp_bucket_ids, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list)

        # log inter attention weights
        if log_inter_attn and (use_hidden_state_attn + use_delta_t_attn + use_week_time_attn > 0) and _batch_number % 100 == 0 and inter_session_seq_length[0] == 15:
            datahandler.log_attention_weights_inter(use_hidden_state_attn, use_delta_t_attn, use_week_time_attn, user_list[0], inter_attn_weights, input_timestamps, dataset)

        
        datahandler.store_user_session_representations(sess_rep, user_list, input_timestamps, input_timestamp_bucket_ids)

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
        
        xinput, targetvalues, sl, input_timestamps, input_timestamp_bucket_ids, session_reps, inter_session_seq_length, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, input_timestamps, input_timestamp_bucket_ids, session_reps, inter_session_seq_length, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list = datahandler.get_next_test_batch()
    intra_rnn.eval()
    inter_rnn.eval()
    embed.eval()
    while len(xinput) > int(BATCH_SIZE / 2):
        batch_start_time = time.time()
        _batch_number += 1

        sess_rep, inter_attn_weights, intra_attn_weights, batch_predictions = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, input_timestamps, input_timestamp_bucket_ids, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list)

        # log intra attention weights
        if log_intra_attn and use_intra_attn and _batch_number % 3 == 0:
            for i in range(len(user_list)):
                if inter_session_seq_length[i] == 15 and sl[i] > 5:
                    datahandler.log_attention_weights_intra(intra_attn_weights, RUN_NAME, sl, batch_predictions, user_list[i], i)

        datahandler.store_user_session_representations(sess_rep, user_list, input_timestamps, input_timestamp_bucket_ids)

        # Evaluate predictions
        tester.evaluate_batch(batch_predictions, targetvalues, sl)

        # Print some stats during testing
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, input_timestamps, input_timestamp_bucket_ids, session_reps, inter_session_seq_length, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall10, current_recall20, mrr5, mrr10, mrr20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    print(test_stats)
    if epoch == 1:
        datahandler.log_config(message)
    datahandler.log_test_stats(epoch, epoch_loss, test_stats)
    tensorboard.scalar_summary('Recall@5', current_recall5, epoch)
    tensorboard.scalar_summary('Recall@10', current_recall10, epoch)
    tensorboard.scalar_summary('Recall@20', current_recall20, epoch)
    tensorboard.scalar_summary('MRR@5', mrr5, epoch)
    tensorboard.scalar_summary('MRR@10', mrr10, epoch)
    tensorboard.scalar_summary('MRR@20', mrr20, epoch)
    tensorboard.scalar_summary('epoch_loss', epoch_loss, epoch)

    epoch += 1

    if save_model_parameters:
        torch.save(embed.state_dict(), HOME + "/savestates/" + RUN_NAME + "-embed_model.pth")
        torch.save(inter_rnn.state_dict(), HOME + "/savestates/" + RUN_NAME + "-inter_model.pth")
        torch.save(intra_rnn.state_dict(), HOME + "/savestates/" + RUN_NAME + "-intra_model.pth")
        torch.save(embed_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-embed_optimizer.pth")
        torch.save(inter_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-inter_optimizer.pth")
        torch.save(intra_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-intra_optimizer.pth")
