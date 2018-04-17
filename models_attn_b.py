import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

class Embed(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embed, self).__init__()
        self.embedding_table = nn.Embedding(input_size, embedding_size)
        self.embedding_table.weight.data.copy_(torch.zeros(input_size,embedding_size).uniform_(-1,1))
        self.embedding_table.weight.data[0] = torch.zeros(embedding_size) #ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output

class InterRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, gpu_no=0):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input, hidden, inter_session_seq_length):
        # gets the output of the last non-zero session representation
        input = self.dropout1(input)
        output, _ = self.gru(input, hidden)

        last_index_of_session_reps = inter_session_seq_length - 1
        hidden_indices = last_index_of_session_reps.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
        hidden_out = torch.gather(output, 1, hidden_indices)
        hidden_out = hidden_out.squeeze()
        hidden_out = hidden_out.unsqueeze(0)
        hidden_out = self.dropout2(hidden_out)
        return output, hidden_out

    # initialize hidden with variable batch size
    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden

class IntraRNN(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_size, n_layers, dropout, max_session_representations, gpu_no=0):
        super(IntraRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, n_items)

    def forward(self, input_embedding, hidden):
        embedded_input = self.dropout1(input_embedding)

        gru_output, hidden = self.gru(embedded_input, hidden)

        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden, embedded_input

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
