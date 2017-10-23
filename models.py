import torch
import torch.nn as nn
from torch.autograd import Variable

class InterRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

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
        return hidden_out

    # initialize hidden with variable batch size
    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        return hidden

class IntraRNN(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_size, n_layers, dropout):
        super(IntraRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_items, embedding_size)
        self.embedding.weight.data.copy_(torch.zeros(n_items, embedding_size).uniform_(-1,1))
        self.embedding.weight.data[0] = torch.zeros(embedding_size) # ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(embedding_size, n_items)

    def forward(self, input, hidden, session_lengths):
        embedded_input = self.embedding(input)
        embedded_input = self.dropout1(embedded_input)
        gru_output, hidden = self.gru(embedded_input, hidden)
        output = self.dropout2(gru_output)
        output = self.linear(output)

        last_index_of_sessions = session_lengths - 1
        hidden_indices = last_index_of_sessions.view(-1, 1, 1).expand(gru_output.size(0), 1, gru_output.size(2))
        hidden_out = torch.gather(gru_output, 1, hidden_indices)
        hidden_out = hidden_out.squeeze()
        hidden_out = hidden_out.unsqueeze(0)

        sum_x = embedded_input.sum(1)
        mean_x = sum_x.div(session_lengths.float())
        
        return output, hidden_out, mean_x

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        return hidden