import datetime
import os
import time
import math
import numpy as np
from lastfm_utils import PlainRNNDataHandler
from test_util import Tester

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from logger import Logger

import sys

reddit = "subreddit"
lastfm = "lastfm"
instacart = "instacart"

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

seed = 1
torch.manual_seed(seed)

N_ITEMS      = -1
BATCHSIZE    = 2

if dataset == reddit:
    INTERNAL_SIZE = 50
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 1.0
elif dataset == lastfm:
    INTERNAL_SIZE = 100
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 0.8
elif dataset == instacart:
    INTERNAL_SIZE = 80
    LEARNING_RATE = 0.001
    DROPOUT_PKEEP = 0.8
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 20-1     # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions. This is important for training, but when running, we can have as long sequences as we want! Just need to keep the hidden state and compute the next action)
EMBEDDING_SIZE = INTERNAL_SIZE
TOP_K = 20
MAX_EPOCHS = 50


# Load training data
datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE, log_file)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

message = "------------------------------------------------------------------------\n"
message += "DATASET: "+dataset+" MODEL: plain RNN"
message += "\nCONFIG: N_ITEMS="+str(N_ITEMS)+" BATCHSIZE="+str(BATCHSIZE)+" INTERNAL_SIZE="+str(INTERNAL_SIZE)
message += "\nN_LAYERS="+str(N_LAYERS)+" SEQLEN="+str(SEQLEN)+" EMBEDDING_SIZE="+str(EMBEDDING_SIZE)
message += "\nN_SESSIONS="+str(N_SESSIONS)+" SEED="+str(seed)+"\n"
datahandler.log_config(message)
print(message)

def to_np(x):
    return x.data.cpu().numpy()

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(MyRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(N_ITEMS, EMBEDDING_SIZE)

        weights = torch.rand(N_ITEMS, EMBEDDING_SIZE)

        weights = weights * 2 - 1

        self.embedding.weight = nn.Parameter(weights)

        self.gru = nn.GRU(EMBEDDING_SIZE, hidden_size, n_layers, dropout=(1-DROPOUT_PKEEP), batch_first=True)

        self.dropout = nn.Dropout(p=(1-DROPOUT_PKEEP))  # the GRU dropout parameter does not add dropout to the final layer

        self.dropout2 = nn.Dropout(p=(1-DROPOUT_PKEEP))  # the GRU dropout parameter does not add dropout to the final layer

        self.linear = nn.Linear(EMBEDDING_SIZE, N_ITEMS)

    def forward(self, input, hidden):

        embedded_input = self.embedding(input)

        #print(embedded_input)

        embedded_input = self.dropout2(embedded_input)

        output, hidden = self.gru(embedded_input, hidden)

        #output = self.dropout(output)

        output = output.contiguous().view(-1, EMBEDDING_SIZE)

        output = self.linear(output)

        return output, hidden

    def initHidden(self):
        hidden = Variable(torch.zeros(self.n_layers, BATCHSIZE, self.hidden_size))   # self.n_layers, batch_size,  self.hidden_size
        if use_cuda:
            return hidden.cuda()
        return hidden


def train(input, target, model, model_optimizer, criterion):
    hidden = model.initHidden()
    model_optimizer.zero_grad()

    input = Variable(torch.LongTensor(input))

    target = Variable(torch.LongTensor(target))

    if use_cuda:
        input = input.cuda()
        target = target.cuda()

    output, hidden = model(input, hidden)

    flatten_target = target.view(-1)

    #loss = criterion(output, flatten_target)

    loss = masked_cross_entropy_loss(output, flatten_target)
    loss = loss.mean(0)

    loss.backward()
    model_optimizer.step()

    return loss

def predict(input, model):
    hidden = model.initHidden()

    input = Variable(torch.LongTensor(input))

    if use_cuda:
        input = input.cuda()

    output, hidden = model(input, hidden)

    Ylogits = output

    top_k = torch.topk(Ylogits, TOP_K)

    top_k_values = top_k[0]
    top_k_predictions = top_k[1]

    Y_prediction = top_k_predictions.view(BATCHSIZE, -1, TOP_K)

    return Y_prediction

model = MyRNN(N_ITEMS, INTERNAL_SIZE, N_LAYERS)   # n_items, hidden_size, n_layers
if use_cuda:
    model = model.cuda()
model_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

logger = Logger('./logs')

#CUSTOM CROSS ENTROPY LOSS(Replace as soon as pytorch has implemented an option for non-summed losses)
#https://github.com/pytorch/pytorch/issues/264
def masked_cross_entropy_loss(y_hat, y):
    logp = -F.log_softmax(y_hat)
    logpy = torch.gather(logp,1,y.view(-1,1))
    mask = Variable(y.data.float().sign().view(-1,1))
    logpy = logpy*mask
    return logpy.view(-1)

##
##  TRAINING
##
print("Starting training.")

epoch = 1
logging_iter = 1

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
        _batch_number = 0
        xinput, targetvalues, sl = datahandler.get_next_train_batch()
        
        while len(xinput) > int(BATCHSIZE/2):
            _batch_number += 1
            batch_start_time = time.time()

            loss = train(xinput, targetvalues, model, model_optimizer, criterion)

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
                logger.scalar_summary('loss', batch_loss, logging_iter)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), logging_iter)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), logging_iter)
                logging_iter += 1
            
            xinput, targetvalues, sl = datahandler.get_next_train_batch()


        print("Epoch", epoch, "finished")
        print("|- Epoch loss:", epoch_loss)
    
    ##
    ##  TESTING
    ##
    print("Starting testing")
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl = datahandler.get_next_test_batch()
    while len(xinput) > int(BATCHSIZE/2):
        batch_start_time = time.time()
        _batch_number += 1

        batch_predictions = predict(xinput, model)
        
        # Evaluate predictions
        tester.evaluate_batch(batch_predictions, targetvalues, sl)

        # Print some stats during testing
        if _batch_number%100==0:
            batch_runtime = time.time() - batch_start_time
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "\t Batch time:", "%.4f" % batch_runtime, "minutes", end='')
            eta = (batch_runtime*(num_test_batches-_batch_number))/60
            eta = "%.2f" % eta
            print("\t ETA:", eta, "minutes.")

        xinput, targetvalues, sl = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print("Recall@5 = " + str(current_recall5))
    print("Recall@20 = " + str(current_recall20))
    logger.scalar_summary('Recall@20', current_recall20,  logging_iter)

    epoch += 1
