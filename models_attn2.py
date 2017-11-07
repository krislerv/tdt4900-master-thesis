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
        self.gru = nn.GRU(embedding_size * 2, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(embedding_size * 2, n_items)

        self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.attn2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, hidden_size)
        self.sss = nn.Softmax()

        self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.v.data.copy_(torch.zeros(1, hidden_size).uniform_(-1, 1))

        self.method = 'concat'
        self.out = nn.Linear(hidden_size * 2, n_items)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 0))
            energy = self.v.dot(energy)
            return energy

    def forward(self, input, last_context, last_hidden, encoder_outputs):
        embedded_input = self.embedding(input)
        embedded_input = self.dropout1(embedded_input)

        rnn_input = torch.cat((embedded_input, last_context), 2)

        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        seq_len = len(encoder_outputs[0])

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(2, seq_len)) # B x 1 x S

        attn_energies = attn_energies.cuda()
        
        hidden_t = hidden.transpose(0, 1).squeeze(1) #.expand(2, 15, 100) # .squeeze(1)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[0, i] = self.score(hidden_t[0], encoder_outputs[0, i])
            attn_energies[1, i] = self.score(hidden_t[1], encoder_outputs[1, i])

        #attn_energies = self.attn(encoder_outputs)
        #attn_energies = hidden_t*attn_energies
        #attn_energies = attn_energies.sum(2)

        #attn_energies = self.attn_combine(torch.cat((hidden_t, encoder_outputs), 2))
        #attn_energies = self.v.dot(attn_energies)
        #print(attn_energies.size())

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        attn_weights = F.softmax(attn_energies).unsqueeze(1)

        attn_weights = attn_weights.cuda()

        context = attn_weights.bmm(encoder_outputs) # B x 1 x N

        output = self.dropout2(rnn_output)

        output = self.out(torch.cat((output, context), 2))

        return output, context, hidden, embedded_input, rnn_output

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        return hidden