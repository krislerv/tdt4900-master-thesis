import datetime
import os
import time
import math
import numpy as np
from lastfm_utils_ii_rnn import IIRNNDataHandler
from test_util import Tester

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from logger import Logger

reddit = "subreddit"
lastfm = "lastfm"
instacart = "instacart"

use_last_hidden_state = True

dataset = lastfm

save_best = True
do_training = True
use_cuda = True

home = os.path.expanduser('~')

# Specify path to dataset here
dataset_path = home + '/datasets/'+dataset+'/4_train_test_split.pickle'
epoch_file = './epoch_file-simple-rnn-'+dataset+'.pickle'
checkpoint_file = './checkpoints/plain-rnn-'+dataset+'-'
checkpoint_file_ending = '.ckpt'
date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-testing-plain-rnn.txt'

seed = 0
torch.manual_seed(seed)

N_ITEMS      = -1
BATCHSIZE    = 2

if dataset == reddit:
    ST_INTERNALSIZE = 50
    LT_INTERNALSIZE = ST_INTERNALSIZE
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 1.0
    MAX_SESSION_REPRESENTATIONS = 15
    MAX_EPOCHS = 31
elif dataset == lastfm:
    ST_INTERNALSIZE = 100
    LT_INTERNALSIZE = ST_INTERNALSIZE
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 0.8
    MAX_SESSION_REPRESENTATIONS = 15
    MAX_EPOCHS = 10 # 50
elif dataset == instacart:
    ST_INTERNALSIZE = 80
    LT_INTERNALSIZE = ST_INTERNALSIZE
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 0.8
    MAX_SESSION_REPRESENTATIONS = 15
    MAX_EPOCHS = 200
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 20-1     # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions. This is important for training, but when running, we can have as long sequences as we want! Just need to keep the hidden state and compute the next action)
EMBEDDING_SIZE = ST_INTERNALSIZE
TOP_K = 20

# Load training data
datahandler = IIRNNDataHandler(dataset_path, BATCHSIZE, log_file, 
        MAX_SESSION_REPRESENTATIONS, LT_INTERNALSIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

message = "------------------------------------------------------------------------\n"
if use_last_hidden_state:
    message += dataset + " with last hidden state\n"
else:
    message += dataset + " with average of embeddings\n"
message += "DATASET: "+dataset+" MODEL: II-RNN"
message += "\nCONFIG: N_ITEMS="+str(N_ITEMS)+" BATCHSIZE="+str(BATCHSIZE)
message += "\nST_INTERNALSIZE="+str(ST_INTERNALSIZE)+" LT_INTERNALSIZE="+str(LT_INTERNALSIZE)
message += "\nN_LAYERS="+str(N_LAYERS)+" SEQLEN="+str(SEQLEN)+" EMBEDDING_SIZE="+str(EMBEDDING_SIZE)
message += "\nN_SESSIONS="+str(N_SESSIONS)+" SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS="+str(MAX_SESSION_REPRESENTATIONS)
message += "\nDROPOUT="+str(DROPOUT_PKEEP)+" LEARNING_RATE="+str(LEARNING_RATE)
datahandler.log_config(message)
print(message)

if not do_training:
    print("\nOBS!!!! Training is turned off!\n")

def to_np(x):
    return x.data.cpu().numpy()

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(MyRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.intergru = nn.GRU(EMBEDDING_SIZE, hidden_size, n_layers)

        self.inter_dropout = nn.Dropout(p=(1-DROPOUT_PKEEP))

        self.embedding = nn.Embedding(N_ITEMS, EMBEDDING_SIZE)

        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size, n_layers, dropout=(1-DROPOUT_PKEEP), batch_first=True)

        self.dropout = nn.Dropout(p=(1-DROPOUT_PKEEP))  # the GRU dropout parameter does not add dropout to the final layer

        self.linear = nn.Linear(EMBEDDING_SIZE, N_ITEMS)

        #self.softmax = nn.Softmax()

    def forward(self, input, hidden, session_reps, inter_session_seq_length):

        session_output, session_hidden = self.intergru(session_reps)

        session_output = self.inter_dropout(session_output)

        initial_hidden_state = torch.gather(session_output, 1, inter_session_seq_length)

        initial_hidden_state = torch.transpose(initial_hidden_state, 0, 1)

        #print(initial_hidden_state)

        embedded_input = self.embedding(input)

        embedded_input = self.dropout(embedded_input)

        output, hidden = self.gru(embedded_input, initial_hidden_state)


        output = output.contiguous().view(-1, EMBEDDING_SIZE)

        output = self.linear(output)

        #output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        hidden = Variable(torch.randn(self.n_layers, BATCHSIZE, self.hidden_size))   # self.n_layers, batch_size,  self.hidden_size
        if use_cuda:
            return hidden.cuda()
        return hidden


def train(input, target, session_reps, inter_session_seq_length, use_last_hidden_state, model, model_optimizer, criterion):
    hidden = model.initHidden()
    model_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))

    target = Variable(torch.LongTensor(target))

    session_reps = Variable(torch.FloatTensor(session_reps))

    inter_session_seq_length = np.asarray(inter_session_seq_length)
    inter_session_seq_length = inter_session_seq_length - 1
    inter_session_seq_length = [[item]*EMBEDDING_SIZE for item in inter_session_seq_length]
    inter_session_seq_length = np.asarray(inter_session_seq_length)
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length).view(BATCHSIZE, 1, ST_INTERNALSIZE))

    if use_cuda:
        input = input.cuda()
        target = target.cuda()
        session_reps = session_reps.cuda()
        inter_session_seq_length = inter_session_seq_length.cuda()

    output, hidden = model(input, hidden, session_reps, inter_session_seq_length)

    flatten_target = target.view(-1)

    loss = criterion(output, flatten_target)

    loss.backward()
    model_optimizer.step()

    return loss, hidden

def predict(input, model, session_reps, inter_session_seq_length):
    hidden = model.initHidden()

    input = Variable(torch.LongTensor(input))

    session_reps = Variable(torch.FloatTensor(session_reps))

    inter_session_seq_length = np.asarray(inter_session_seq_length)
    inter_session_seq_length = inter_session_seq_length - 1
    inter_session_seq_length = [[item]*EMBEDDING_SIZE for item in inter_session_seq_length]
    inter_session_seq_length = np.asarray(inter_session_seq_length)
    inter_session_seq_length = Variable(torch.LongTensor(inter_session_seq_length).view(BATCHSIZE, 1, ST_INTERNALSIZE))

    if use_cuda:
        input = input.cuda()
        session_reps = session_reps.cuda()
        inter_session_seq_length = inter_session_seq_length.cuda()


    output, hidden = model(input, hidden, session_reps, inter_session_seq_length)

    Ylogits = output

    top_k = torch.topk(Ylogits, TOP_K)

    top_k_values = top_k[0]
    top_k_predictions = top_k[1]

    Y_prediction = top_k_predictions.view(BATCHSIZE, -1, TOP_K)

    return Y_prediction


model = MyRNN(N_ITEMS, ST_INTERNALSIZE, N_LAYERS)   # n_items, hidden_size, n_layers
if use_cuda:
    model = model.cuda()
model_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

logger = Logger('./logs')

##
##  TRAINING
##
print("Starting training.")

epoch = 1

print()

best_recall5 = -1
best_recall20 = -1

num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
while epoch <= MAX_EPOCHS:
    print("Starting epoch #"+str(epoch))
    epoch_loss = 0

    if do_training:
        datahandler.reset_user_batch_data()
        datahandler.reset_user_session_representations()
        _batch_number = 0
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, _ = datahandler.get_next_train_batch()

        while len(xinput) > int(BATCHSIZE/2):
            _batch_number += 1
            batch_start_time = time.time()
            loss, sess_rep = train(xinput, targetvalues, session_reps, inter_session_seq_length, use_last_hidden_state, model, model_optimizer, criterion)

            datahandler.store_user_session_representations(sess_rep.data.select(0,0).cpu().tolist(), user_list)

            batch_loss = loss.data[0]

            epoch_loss += batch_loss
            if _batch_number%100==0:
                batch_runtime = time.time() - batch_start_time
                print("Batch number:", str(_batch_number), "/", str(num_training_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
                print("\t Batch loss:", "%.3f" % batch_loss, end='')
                eta = (batch_runtime * (num_training_batches-_batch_number)) / 60
                eta = "%.2f" % eta
                print("\t ETA:", eta, "minutes.")

                #============ TensorBoard logging ============#
                logger.scalar_summary('loss', batch_loss, 10000*(epoch) + _batch_number)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), 10000*(epoch) + _batch_number)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), 10000*(epoch) + _batch_number)
            
            xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, _ = datahandler.get_next_train_batch()

        print("Epoch", epoch, "finished")
        print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, _ = datahandler.get_next_test_batch()
    while len(xinput) > int(BATCHSIZE/2):
        batch_start_time = time.time()
        _batch_number += 1

        batch_predictions = predict(xinput, model, session_reps, inter_session_seq_length)
        
        # Evaluate predictions
        tester.evaluate_batch(batch_predictions, targetvalues, sl)

        # Print some stats during testing
        if _batch_number%100==0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number))/60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, session_reps, inter_session_seq_length, user_list, _ = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    logger.scalar_summary('Recall@20', current_recall20,  10000*(epoch) + _batch_number)

    epoch += 1
