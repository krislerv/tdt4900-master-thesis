import datetime
import os
import time
import numpy as np
from models_attn_b import InterRNN, IntraRNN, Embed, OnTheFlySessionRepresentations
from datahandler_attn_b import IIRNNDataHandler
from test_util_h import Tester

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from tensorboard import Logger as TensorBoard

import gpustat

# datasets
reddit = "subreddit"
lastfm = "lastfm-full"
dataset = reddit

# GPU settings
use_cuda = True
GPU_NO = 0  # Dont touch! change CUDA_VISIBLE_DEVICES instead

CUDA_VISIBLE_DEVICES = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

PID = str(os.getpid())

method_on_the_fly = "ATTN-G"  # AVG, LHS, ATTN-G, ATTN-L
method_inter = "ATTN-G"
use_delta_t_attn = False
bidirectional = False
attention_on = "output" # input, output

# logging of attention weights
log_on_the_fly_attn = False
log_inter_attn = False

skip_early_testing = True

# saving/loading of model parameters
save_model_parameters = True
resume_model = False
resume_model_name = "2018-06-09-00-33-54-hierarchical-subreddit"    # unused if resume_model is False

if resume_model:
    skip_early_testing = False


# dataset path
HOME = ".."
DATASET_PATH = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

# logging of testing results
DATE_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
TIME_NOW = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
if resume_model:
    RUN_NAME = resume_model_name
else:
    RUN_NAME = str(DATE_NOW) + '-' + str(TIME_NOW) + '-hierarchical-' + dataset
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
message += dataset + " with average of embeddings\n"
message += "DATASET: " + dataset + " MODEL: attn-RNN"
message += "\nCONFIG: N_ITEMS=" + str(N_ITEMS) + " BATCH_SIZE=" + str(BATCH_SIZE)
message += "\nINTRA_INTERNAL_SIZE=" + str(INTRA_INTERNAL_SIZE) + " INTER_INTERNAL_SIZE=" + str(INTER_INTERNAL_SIZE)
message += "\nN_LAYERS=" + str(N_LAYERS) + " EMBEDDING_SIZE=" + str(EMBEDDING_SIZE)
message += "\nN_SESSIONS=" + str(N_SESSIONS) + " SEED="+str(seed) + " GPU_NO=" + str(GPU_NO) + " (" + CUDA_VISIBLE_DEVICES + ")" + " PID=" + PID
message += "\nMAX_SESSION_REPRESENTATIONS=" + str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT_RATE=" + str(DROPOUT_RATE) + " LEARNING_RATE=" + str(LEARNING_RATE)
message += "\nmethod_inter=" + method_inter + " method_on_the_fly=" + method_on_the_fly + " use_delta_t_attn=" + str(use_delta_t_attn) + " attention_on=" + attention_on
message += "\nbidirectional=" + str(bidirectional)
if resume_model:
    message += "\nresume_model: " + resume_model_name
print(message)

def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))

embed = Embed(N_ITEMS, EMBEDDING_SIZE)
if use_cuda:
    embed = embed.cuda(GPU_NO)
embed_optimizer = optim.Adam(embed.parameters(), lr=LEARNING_RATE)

# initialize inter RNN
inter_rnn = InterRNN(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, method_inter, method_on_the_fly, use_delta_t_attn, bidirectional, attention_on, gpu_no=GPU_NO)
if use_cuda:
    inter_rnn = inter_rnn.cuda(GPU_NO)
inter_optimizer = optim.Adam(inter_rnn.parameters(), lr=LEARNING_RATE)

# initialize intra RNN
intra_rnn = IntraRNN(N_ITEMS, INTRA_INTERNAL_SIZE, EMBEDDING_SIZE, N_LAYERS, DROPOUT_RATE, MAX_SESSION_REPRESENTATIONS, bidirectional, gpu_no=GPU_NO)
if use_cuda:
    intra_rnn = intra_rnn.cuda(GPU_NO)
intra_optimizer = optim.Adam(intra_rnn.parameters(), lr=LEARNING_RATE)

on_the_fly_sess_reps = OnTheFlySessionRepresentations(EMBEDDING_SIZE, INTER_INTERNAL_SIZE, N_LAYERS, DROPOUT_RATE, method_on_the_fly, bidirectional, attention_on, gpu_no=GPU_NO)
if use_cuda:
    on_the_fly_sess_reps = on_the_fly_sess_reps.cuda(GPU_NO)
on_the_fly_sess_reps_optimizer = optim.Adam(on_the_fly_sess_reps.parameters(), lr=LEARNING_RATE)

if resume_model:
    embed.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-embed_model.pth"))
    embed_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-embed_optimizer.pth"))
    inter_rnn.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-inter_model.pth"))
    inter_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-inter_optimizer.pth"))
    intra_rnn.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-intra_model.pth"))
    intra_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-intra_optimizer.pth"))
    on_the_fly_sess_reps.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-on_the_fly_sess_reps_model.pth"))
    on_the_fly_sess_reps_optimizer.load_state_dict(torch.load(HOME + "/savestates/" + RUN_NAME + "-on_the_fly_sess_reps_optimizer.pth"))

def run(input, target, session_lengths, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps):
    if intra_rnn.training:
        inter_optimizer.zero_grad()
        intra_optimizer.zero_grad()
        embed_optimizer.zero_grad()
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
    input_timestamps = Variable(torch.FloatTensor(input_timestamps))
    previous_session_timestamps = Variable(torch.FloatTensor(previous_session_timestamps))

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
        input_timestamps = input_timestamps.cuda(GPU_NO)
        previous_session_timestamps = previous_session_timestamps.cuda(GPU_NO)

    input_embedding = embed(input)
    input_embedding = F.dropout(input_embedding, DROPOUT_RATE, intra_rnn.training, False)

    all_session_representations = Variable(torch.zeros(input.size(0), MAX_SESSION_REPRESENTATIONS, (1 + (bidirectional and not method_on_the_fly == "AVG")) * INTER_INTERNAL_SIZE)).cuda(GPU_NO)
    on_the_fly_attn_weights = Variable(torch.zeros(input.size(0), MAX_SESSION_REPRESENTATIONS, 20))

    for i in range(input.size(0)):
        user_previous_session_batch = previous_session_batch[i]
        user_previous_session_lengths = previous_session_lengths[i]
        user_prevoius_session_counts = prevoius_session_counts[i]

        user_previous_session_batch_embedding = embed(user_previous_session_batch)
        user_previous_session_batch_embedding = F.dropout(user_previous_session_batch_embedding, DROPOUT_RATE, intra_rnn.training, False)

        hidden = on_the_fly_sess_reps.init_hidden(MAX_SESSION_REPRESENTATIONS, use_cuda=use_cuda)

        all_session_representations[i], on_the_fly_attn_weights[i] = on_the_fly_sess_reps(hidden, user_previous_session_batch_embedding, user_previous_session_lengths, user_prevoius_session_counts, user_list[i])

    input_timestamps = input_timestamps.unsqueeze(1).expand(input.size(0), MAX_SESSION_REPRESENTATIONS)
    delta_t = input_timestamps - previous_session_timestamps
    delta_t_hours = delta_t.div(3600).floor().long()
    delta_t_ceiling = Variable(torch.LongTensor([168]).expand(input.size(0), MAX_SESSION_REPRESENTATIONS)).cuda(GPU_NO)
    delta_t_hours = torch.min(delta_t_hours, delta_t_ceiling)

    inter_hidden = inter_rnn.init_hidden(session_reps.size(0), use_cuda)
    inter_hidden, inter_attn_weights = inter_rnn(all_session_representations, inter_hidden, prevoius_session_counts, user_list, delta_t_hours)

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
        on_the_fly_sess_reps_optimizer.step()

    # get average pooling of input for session representations
    sum_x = input_embedding_d.sum(1)
    mean_x = sum_x.div(session_lengths.float())

    top_k_values, top_k_predictions = torch.topk(output, TOP_K)

    # return loss and new session representation
    if intra_rnn.training:
        return loss.data[0], mean_x.data, top_k_predictions, inter_attn_weights, on_the_fly_attn_weights
    else:
        return mean_x.data, top_k_predictions, inter_attn_weights, on_the_fly_attn_weights

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

    if epoch == 1:
        datahandler.log_config(message)

    datahandler.reset_user_batch_data()
    datahandler.reset_user_session_representations()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps = datahandler.get_next_train_batch()
    intra_rnn.train()
    inter_rnn.train()
    embed.train()
    on_the_fly_sess_reps.train()
    while len(xinput) > int(BATCH_SIZE / 2):
        _batch_number += 1
        batch_start_time = time.time()

        batch_loss, sess_rep, top_k_predictions, inter_attn_weights, on_the_fly_attn_weights = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps)

        if log_on_the_fly_attn:
            datahandler.log_attention_weights_on_the_fly(RUN_NAME, user_list[0], on_the_fly_attn_weights[0], previous_session_batch[0])
        
        # log inter attention weights
        if log_inter_attn:
            datahandler.log_attention_weights_inter(RUN_NAME, user_list[0], inter_attn_weights, input_timestamps)
        
        
        datahandler.store_user_session_representations(sess_rep, user_list)

        epoch_loss += batch_loss
        if _batch_number % 100 == 0:
            batch_runtime = time.time() - batch_start_time
            print("PID:", PID, "\t Batch number:", str(_batch_number), "/", str(num_training_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            print("\t Batch loss:", "%.3f" % batch_loss, end='')
            eta = (batch_runtime * (num_training_batches - _batch_number)) / 60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")

        #============ TensorBoard logging ============#
        tensorboard.scalar_summary('batch_loss', batch_loss, log_count)
        log_count += 1
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)

    if (dataset == lastfm and epoch >= 10) or (dataset == reddit and epoch >= 4) or not skip_early_testing:
    
        ##
        ##  TESTING
        ##
        print("Starting testing")
        tester = Tester()
        datahandler.reset_user_batch_data()
        _batch_number = 0
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps = datahandler.get_next_test_batch()
        intra_rnn.eval()
        inter_rnn.eval()
        embed.eval()
        on_the_fly_sess_reps.eval()
        while len(xinput) > int(BATCH_SIZE / 2):
            batch_start_time = time.time()
            _batch_number += 1

            sess_rep, batch_predictions, inter_attn_weights, on_the_fly_attn_weights = run(xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps)

            datahandler.store_user_session_representations(sess_rep, user_list)

            # Evaluate predictions
            tester.evaluate_batch(batch_predictions, targetvalues, sl)

            # Print some stats during testing
            if _batch_number % 100 == 0:
                batch_runtime = time.time() - batch_start_time
                print("PID:", PID, "\t Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
                eta = (batch_runtime*(num_test_batches-_batch_number)) / 60
                eta = "%.2f" % eta
                print("\t ETA:", eta, "minutes.")
            
            xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, previous_session_batch, previous_session_lengths, prevoius_session_counts, input_timestamps, previous_session_timestamps = datahandler.get_next_test_batch()

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
        torch.save(on_the_fly_sess_reps.state_dict(), HOME + "/savestates/" + RUN_NAME + "-on_the_fly_sess_reps_model.pth")
        torch.save(embed_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-embed_optimizer.pth")
        torch.save(inter_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-inter_optimizer.pth")
        torch.save(intra_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-intra_optimizer.pth")
        torch.save(on_the_fly_sess_reps_optimizer.state_dict(), HOME + "/savestates/" + RUN_NAME + "-on_the_fly_sess_reps_optimizer.pth")
