import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class InterRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, use_hidden_state_attn=False, use_delta_t_attn=False, use_week_time_attn=False):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.use_hidden_state_attn = use_hidden_state_attn
        self.use_delta_t_attn = use_delta_t_attn
        self.use_week_time_attn = use_week_time_attn

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if use_delta_t_attn:
            self.delta_embedding = nn.Embedding(169, embedding_size)
            self.delta_embedding.weight.data.copy_(torch.zeros(169, embedding_size).uniform_(-1, 1))

        if use_week_time_attn:
            self.timestamp_embedding = nn.Embedding(169, embedding_size)
            self.timestamp_embedding.weight.data.copy_(torch.zeros(169, embedding_size).uniform_(-1, 1))

        self.num_types_attn = use_hidden_state_attn + use_delta_t_attn + use_week_time_attn

        if self.num_types_attn:  # if at least one type of attention
            self.attention = nn.Linear(self.hidden_size * self.num_types_attn, self.hidden_size)
            self.c = nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.c.data.copy_(torch.zeros(1, hidden_size).uniform_(-1, 1))

            self.index_list = []
            for i in range(max_session_representations):
                self.index_list.append(i)
            self.index_list = torch.LongTensor(self.index_list).cuda()

    def forward(self, input, hidden, inter_session_seq_length, delta_t_h, timestamps):
        # gets the output of the last non-zero session representation
        input = self.dropout1(input)
        output, _ = self.gru(input, hidden)

        if not self.num_types_attn:  # if no attention is to be used
            last_index_of_session_reps = inter_session_seq_length - 1
            hidden_indices = last_index_of_session_reps.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            hidden_out = torch.gather(output, 1, hidden_indices)
            hidden_out = hidden_out.squeeze()
            hidden_out = hidden_out.unsqueeze(0)
            hidden_out = self.dropout2(hidden_out)
            return output, hidden_out, []

        # create a mask so that attention weights for "empty" session representations are zero. Little impact on accuracy, but makes visualization prettier
        inter_session_seq_length = inter_session_seq_length.unsqueeze(1).expand(input.size(0), self.max_session_representations)
        indexes = self.index_list.unsqueeze(0).expand(input.size(0), self.max_session_representations)    # [BATCH_SIZE, MAX_SESS_REP]
        mask = torch.lt(indexes, inter_session_seq_length) # [BATCH_SIZE, MAX_SESS_REP]  i-th element in each batch is 1 if it is a real sess rep, 0 otherwise
        mask = mask.unsqueeze(2).expand(input.size(0), self.max_session_representations, self.hidden_size).float() * 1000000 - 999999   # 1 -> 1, 0 -> -999999

        # create concatenation of the different attention variables.
        first_attn_variable = True # whether concatenated_attention has been assigned to another attention variable (in which case, we should concatenate)

        if self.use_hidden_state_attn:
            concatenated_attention = output
            first_attn_variable = False
        if self.use_delta_t_attn:
            delta_t_h = self.delta_embedding(delta_t_h)
            if first_attn_variable:
                concatenated_attention = delta_t_h
            else:
                concatenated_attention = torch.cat((concatenated_attention, delta_t_h), 2)
            first_attn_variable = False
        if self.use_week_time_attn:
            timestamps = self.timestamp_embedding(timestamps)
            if first_attn_variable:
                concatenated_attention = timestamps
            else:
                concatenated_attention = torch.cat((concatenated_attention, timestamps), 2)
            first_attn_variable = False

        attention_energies = torch.tanh(self.attention(concatenated_attention))
        attention_energies = attention_energies * mask
        attention_energies = attention_energies.transpose(1, 2)
        attention_energies = torch.bmm(self.c.expand(input.size(0), 1, self.hidden_size), attention_energies)
        inter_output_attn_weights = F.softmax(attention_energies.squeeze())
        new_hidden = torch.bmm(inter_output_attn_weights.unsqueeze(1), output)
        new_hidden = new_hidden.transpose(0, 1)
        new_hidden = self.dropout2(new_hidden)

        return output, new_hidden, inter_output_attn_weights

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

    def forward(self, input, hidden, inter_output):
        embedded_input = self.embedding(input)
        embedded_input = self.dropout1(embedded_input)
        gru_output, hidden = self.gru(embedded_input, hidden)
        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden, embedded_input, gru_output

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        return hidden
