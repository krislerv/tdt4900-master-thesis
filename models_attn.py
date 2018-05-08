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
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, bidirectional, use_hidden_state_attn=False, use_delta_t_attn=False, use_week_time_attn=False, gpu_no=0):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.bidirectional = bidirectional

        self.use_hidden_state_attn = use_hidden_state_attn
        self.use_delta_t_attn = use_delta_t_attn
        self.use_week_time_attn = use_week_time_attn

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        if use_delta_t_attn:
            self.delta_embedding = nn.Embedding(169, embedding_size)
            self.delta_embedding.weight.data.copy_(torch.zeros(169, embedding_size).uniform_(-1, 1))

        if use_week_time_attn:
            self.timestamp_embedding = nn.Embedding(169, embedding_size)
            self.timestamp_embedding.weight.data.copy_(torch.zeros(169, embedding_size).uniform_(-1, 1))

        self.num_types_attn = use_hidden_state_attn + use_delta_t_attn + use_week_time_attn

        if self.num_types_attn:  # if at least one type of attention
            self.attention = nn.Linear(self.hidden_size * (self.num_types_attn + (self.use_hidden_state_attn and self.bidirectional)), self.hidden_size)
            self.c = nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.c.data.copy_(torch.zeros(1, hidden_size).uniform_(-1, 1))

            self.index_list = []
            for i in range(max_session_representations):
                self.index_list.append(i)
            self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)

    def forward(self, input, hidden, inter_session_seq_length, delta_t_h, timestamps):
        # gets the output of the last non-zero session representation
        input = self.dropout(input)
        output, _ = self.gru(input, hidden)

        if not self.num_types_attn:  # if no attention is to be used
            last_index_of_session_reps = inter_session_seq_length - 1
            hidden_indices = last_index_of_session_reps.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            hidden_out = torch.gather(output, 1, hidden_indices)
            hidden_out = hidden_out.squeeze()
            hidden_out = hidden_out.unsqueeze(0)
            hidden_out = self.dropout(hidden_out)
            return output, hidden_out, []

        # create a mask so that attention weights for "empty" session representations are zero. Little impact on accuracy, but makes visualization prettier
        inter_session_seq_length = inter_session_seq_length.unsqueeze(1).expand(input.size(0), self.max_session_representations)
        indexes = self.index_list.unsqueeze(0).expand(input.size(0), self.max_session_representations)    # [BATCH_SIZE, MAX_SESS_REP]
        mask = torch.lt(indexes, inter_session_seq_length) # [BATCH_SIZE, MAX_SESS_REP]  i-th element in each batch is 1 if it is a real sess rep, 0 otherwise
        mask = mask.unsqueeze(2).expand(input.size(0), self.max_session_representations, self.hidden_size * (self.num_types_attn + (self.use_hidden_state_attn and self.bidirectional))).float()

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

        concatenated_attention = concatenated_attention * mask

        attention_energies = torch.tanh(self.attention(concatenated_attention))
        attention_energies = attention_energies.transpose(1, 2)
        attention_energies = torch.bmm(self.c.expand(input.size(0), 1, self.hidden_size), attention_energies)
        inter_output_attn_weights = F.softmax(attention_energies.squeeze())
        new_hidden = torch.bmm(inter_output_attn_weights.unsqueeze(1), output)
        new_hidden = new_hidden.transpose(0, 1)
        new_hidden = self.dropout(new_hidden)

        return output, new_hidden, inter_output_attn_weights

    # initialize hidden with variable batch size
    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros((1 + self.bidirectional) * self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden

class IntraRNN(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_size, n_layers, dropout, max_session_representations, bidirectional, use_attn=False, use_per_user_intra_attn=False, intra_attn_method="cat", gpu_no=0):
        super(IntraRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.bidirectional = bidirectional

        self.use_attn = use_attn
        self.attn_method = intra_attn_method
        self.use_per_user_intra_attn = use_per_user_intra_attn

        self.gru = nn.GRU((2 + self.bidirectional) * embedding_size if use_attn else embedding_size, (1 + bidirectional) * hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear((1 + bidirectional) * hidden_size, n_items)

        if self.use_attn:
            self.hidden_attention = nn.Linear((1 + self.bidirectional) * self.hidden_size, self.hidden_size, bias=False)
            self.inter_output_attention = nn.Linear((1 + self.bidirectional) * self.hidden_size, self.hidden_size, bias=False)
            self.scale = nn.Linear(self.hidden_size, 1, bias=False)
            self.cat_attention = nn.Linear((1 + self.bidirectional) * 2 * hidden_size, hidden_size)

            if use_per_user_intra_attn:
                self.inter_params = nn.ModuleList([nn.Linear((1 + self.bidirectional) * hidden_size, hidden_size) for i in range(1000)])
                self.hidden_params = nn.ModuleList([nn.Linear((1 + self.bidirectional) * hidden_size, hidden_size) for i in range(1000)])
                self.scale_params = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(1000)])
                self.cat_attention_params = nn.ModuleList([nn.Linear((1 + self.bidirectional) * 2 * hidden_size, hidden_size) for i in range(1000)])

        if self.bidirectional:
            self.gru_scale = nn.Linear(2 * self.hidden_size, self.hidden_size)



    def forward(self, input_embedding, hidden, inter_output, delta_t_h, user_list):
        embedded_input = self.dropout(input_embedding)

        if self.use_attn:
            hidden_t = hidden.transpose(0, 1)

            ### per user attention weights
            if self.use_per_user_intra_attn:
                if self.attn_method == "cat":
                    result = Variable(torch.zeros(input_embedding.size(0), self.max_session_representations, self.hidden_size)).cuda(self.gpu_no)
                    for i in range(input_embedding.size(0)):
                        result[i] = torch.tanh(self.cat_attention_params[user_list[i].data[0]](torch.cat((hidden_t[i].expand(self.max_session_representations, (1 + self.bidirectional) * self.hidden_size), inter_output[i]), dim=1)))
                elif self.attn_method == "sum":
                    result = Variable(torch.zeros(input_embedding.size(0), self.max_session_representations, self.hidden_size)).cuda(self.gpu_no)
                    for i in range(input_embedding.size(0)):
                        user_inter_output = self.inter_params[user_list[i].data[0]](inter_output[i])
                        user_hidden_t = self.hidden_params[user_list[i].data[0]](hidden_t[i])
                        result[i] = torch.tanh(user_hidden_t.expand(self.max_session_representations, self.hidden_size) + user_inter_output)                      # sum last hidden and inter output
                else:
                    raise Exception("Invalid method")
                energies = Variable(torch.zeros(input_embedding.size(0), self.max_session_representations, 1)).cuda(self.gpu_no)
                for i in range(input_embedding.size(0)):
                    energies[i] = self.scale_params[user_list[i].data[0]](result[i])

            ### global attention weights
            else:
                if self.attn_method == "cat":
                    result = torch.tanh(self.cat_attention(torch.cat((hidden_t.expand(input_embedding.size(0), self.max_session_representations, (1 + self.bidirectional) * self.hidden_size), inter_output), dim=2)))  # concat last hidden and inter output
                elif self.attn_method == "sum":
                    hidden_t_a = self.hidden_attention(hidden_t)
                    inter_output_a = self.inter_output_attention(inter_output)
                    result = torch.tanh(hidden_t_a.expand(input_embedding.size(0), self.max_session_representations, self.hidden_size) + inter_output_a)                      # sum last hidden and inter output
                else:
                    raise Exception("Invalid method")
                energies = self.scale(result)
            
            attn_weights = F.softmax(energies.squeeze(), dim=1)

            context = torch.bmm(attn_weights.unsqueeze(1), inter_output)
            gru_input = torch.cat((embedded_input, context), 2)
            gru_output, hidden = self.gru(gru_input, hidden)
        else:
            gru_output, hidden = self.gru(embedded_input, hidden)
            attn_weights = []

        output = self.dropout(gru_output)
        output = self.linear(output)

        if self.bidirectional:  # if using LHS session representations, make them the correct size
            gru_output = self.gru_scale(gru_output)

        return output, hidden, embedded_input, gru_output, attn_weights

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros((1 + self.bidirectional) * self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
