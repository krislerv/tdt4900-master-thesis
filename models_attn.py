import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return output, hidden_out

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
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(n_items, embedding_size)
        self.embedding.weight.data.copy_(torch.zeros(n_items, embedding_size).uniform_(-1, 1))
        self.embedding.weight.data[0] = torch.zeros(embedding_size) # ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(embedding_size, n_items)

        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, hidden_size)
        self.sss = nn.Softmax()

        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.v.data.copy_(torch.zeros(1, hidden_size).uniform_(-1, 1))


    def forward(self, input, hidden, inter_output):
        embedded_input = self.embedding(input)
        embedded_input = self.dropout1(embedded_input)

        print(embedded_input.size())

        
        attention_weights = []

        for i in range(inter_output.size(1)):
            a = Variable(torch.LongTensor([i]).expand(inter_output.size(0), 1, self.embedding_size))
            a = a.cuda()
            b = torch.gather(inter_output, 1, a)

            #nn = torch.tanh(b + hidden.transpose(0, 1))
            nn = torch.tanh(self.attn(torch.cat((b, hidden.transpose(0, 1)), 2)))

            nn = self.v.expand(2, 1, 100)*nn

            nn = nn.sum(2)

            if i == 0:
                attention_weights = nn
            else:
                attention_weights = torch.cat((attention_weights, nn), 1)
        attention_weights = F.softmax(attention_weights)

        attn_applied = attention_weights.unsqueeze(2)*inter_output
        context_vector = attn_applied.sum(1).unsqueeze(1)

        #embedded_input = embedded_input + context_vector
        embedded_input = self.attn_combine(torch.cat((embedded_input, context_vector), 2))
        
        

        """
        a = torch.cat((embedded_input, hidden.transpose(0, 1)), 2)
        b = self.attn(a)
        attn_weights = self.sss(b.squeeze()).unsqueeze(1)
        attn_applied = torch.bmm(attn_weights, inter_output)
        output = torch.cat((embedded_input, attn_applied), 2)
        output = self.attn_combine(output)
        #output = F.relu(output)
        gru_output, hidden = self.gru(output, hidden)
        """
        

        gru_output, hidden = self.gru(embedded_input, hidden)
        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden, embedded_input, gru_output

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        return hidden