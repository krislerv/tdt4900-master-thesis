import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

class Embed(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Embed, self).__init__()
        self.embedding_table = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.embedding_table.weight.data.copy_(torch.zeros(input_size,embedding_size).uniform_(-1,1))
        self.embedding_table.weight.data[0] = torch.zeros(embedding_size) #ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output

class SessRepEmbed(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(SessRepEmbed, self).__init__()
        self.embedding_table = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.embedding_table.weight.data.copy_(torch.zeros(input_size,embedding_size).uniform_(-1,1))
        self.embedding_table.weight.data[0] = torch.zeros(embedding_size) #ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
    
    def forward(self, input):
        output = self.embedding_table(input)
        return output

class OnTheFlySessionRepresentations(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, method, gpu_no=0):
        super(OnTheFlySessionRepresentations, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.method = method

        self.gpu_no = gpu_no

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)

        self.attention = nn.Linear(hidden_size, hidden_size)
        self.scale = nn.Linear(hidden_size, 1, bias=False)

        self.index_list = []
        for i in range(20):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)

        if method == "ATTN-L":
            self.user_attention = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1000)])
            self.user_scale = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(1000)])

    def forward(self, hidden, user_previous_session_batch_embedding, user_previous_session_lengths, user_prevoius_session_counts, user_id):
        if self.method == "LHS":
            output, hidden = self.gru(user_previous_session_batch_embedding, hidden)

            # gets the last actual hidden state (generated from a non-zero input)
            hidden_indices = user_previous_session_lengths.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
            hidden = torch.gather(output, 1, hidden_indices)
            hidden = hidden.squeeze()
            hidden = self.dropout(hidden)

            return hidden   # [MAX_SESS_REP x EMBEDDING SIZE]

        elif self.method == "AVG":
            # padding sessions have length 0, make them length 1 to avoid division by 0 error
            zeros = Variable(torch.zeros(user_previous_session_lengths.size(0))).long().cuda(self.gpu_no)
            is_padding_session = torch.eq(zeros, user_previous_session_lengths).long()
            user_previous_session_lengths = user_previous_session_lengths + is_padding_session

            user_previous_session_batch_embedding_summed = user_previous_session_batch_embedding.sum(1)
            mean_user_previous_session_batch_embedding = user_previous_session_batch_embedding_summed.transpose(0, 1).div(user_previous_session_lengths.float()).transpose(0, 1)

            return mean_user_previous_session_batch_embedding   # [MAX_SESS_REP x EMBEDDING SIZE]

        elif self.method == "ATTN-G" or self.method == "ATTN-L":
            output, hidden = self.gru(user_previous_session_batch_embedding, hidden)

            # create a mask so that attention weights for "empty" outputs are zero
            user_previous_session_lengths_expanded = user_previous_session_lengths.unsqueeze(1).expand(output.size(0), output.size(1))     # [MAX_SESS_REP, MAX_SEQ_LEN]
            indexes = self.index_list.unsqueeze(0).expand(output.size(0), output.size(1))                                      # [MAX_SESS_REP, MAX_SEQ_LEN]
            mask = torch.le(indexes, user_previous_session_lengths_expanded) # [MAX_SESS_REP, MAX_SEQ_LEN]  i-th element in each batch is 1 if it is a real item, 0 otherwise
            mask = mask.unsqueeze(2).expand(output.size(0), output.size(1), output.size(2)).float() * 1000000 - 1000000   # 1 -> 0, 0 -> -1000000    [MAX_SESS_REP, MAX_SEQ_LEN, HIDDEN_SIZE]

            # apply mask
            output = output + mask

            # compute attention weights
            if self.method == "ATTN-G":
                attn_energies = torch.tanh(self.attention(output))
                attn_energies = self.scale(attn_energies)
                attn_weights = F.softmax(attn_energies.squeeze(), dim=1)
            else:
                attn_energies = torch.tanh(self.user_attention[user_id](output))
                attn_energies = self.user_scale[user_id](attn_energies)
                attn_weights = F.softmax(attn_energies.squeeze(), dim=1)

            # apply attention weights
            session_representations = torch.bmm(attn_weights.unsqueeze(1), user_previous_session_batch_embedding)

            session_representations = self.dropout(session_representations.squeeze())

            return session_representations

        else:
            raise Exception("Invalid method")

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
        

class InterRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, method, gpu_no=0):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.method = method

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attention = nn.Linear(hidden_size, hidden_size)
        self.scale = nn.Linear(hidden_size, 1, bias=False)

        self.index_list = []
        for i in range(self.max_session_representations):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)

        if method == "ATTN-L":
            self.user_attention = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1000)])
            self.user_scale = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(1000)])

    def forward(self, all_session_representations, hidden, previous_session_counts, user_list):
        if self.method == "LHS":
            all_session_representations = self.dropout1(all_session_representations)
            output, _ = self.gru(all_session_representations, hidden)

            # since we are subtracting all previous_session_counts by one to get the last index of a real session representation,
            # we need make sure that users with no previous sessions don't get -1 as their last index
            zeros = Variable(torch.zeros(previous_session_counts.size(0))).long().cuda(self.gpu_no)
            has_no_previous_sessions = torch.eq(zeros, previous_session_counts).long()

            last_index_of_session_reps = previous_session_counts + has_no_previous_sessions - 1
            hidden_indices = last_index_of_session_reps.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))

            hidden_out = torch.gather(output, 1, hidden_indices)
            hidden_out = hidden_out.transpose(0, 1)
            hidden_out = self.dropout2(hidden_out)

            return hidden_out

        elif self.method == "AVG":
            # users with no previous sessions have session count 0, make them count 1 to avoid division by 0 error
            zeros = Variable(torch.zeros(previous_session_counts.size(0))).long().cuda(self.gpu_no)
            has_no_previous_sessions = torch.eq(zeros, previous_session_counts).long()
            previous_session_counts = previous_session_counts + has_no_previous_sessions

            all_session_representations_summed = all_session_representations.sum(1)
            mean_all_session_representations_summed = all_session_representations_summed.transpose(0, 1).div(previous_session_counts.float()).transpose(0, 1)

            return mean_all_session_representations_summed.unsqueeze(0).contiguous()

        elif self.method == "ATTN-G" or self.method == "ATTN-L":
            all_session_representations = self.dropout1(all_session_representations)
            output, _ = self.gru(all_session_representations, hidden)
            # create a mask so that attention weights for "empty" outputs are zero
            previous_session_counts_expanded = previous_session_counts.unsqueeze(1).expand(output.size(0), output.size(1))     # [BATCH_SIZE, MAX_SESS_REP]
            indexes = self.index_list.unsqueeze(0).expand(output.size(0), output.size(1))                                      # [BATCH_SIZE, MAX_SESS_REP]
            mask = torch.lt(indexes, previous_session_counts_expanded) # [BATCH_SIZE, MAX_SESS_REP]  i-th element in each batch is 1 if it is a real sess rep, 0 otherwise
            mask = mask.unsqueeze(2).expand(output.size(0), output.size(1), output.size(2)).float() * 1000000 - 1000000   # 1 -> 0, 0 -> -1000000    [BATCH_SIZE, MAX_SESS_REP, HIDDEN_SIZE]

            # apply mask
            output = output + mask

            # compute attention weights
            if self.method == "ATTN-G":
                attn_energies = torch.tanh(self.attention(output))
                attn_energies = self.scale(attn_energies)
                attn_weights = F.softmax(attn_energies.squeeze(), dim=1)
            else:
                attn_energies = Variable(torch.zeros(all_session_representations.size(0), all_session_representations.size(1), 1)).cuda(self.gpu_no)
                for i in range(len(user_list)):
                    user_id = user_list[i]
                    user_attn_energies = torch.tanh(self.user_attention[user_id](output[i]))
                    user_attn_energies = self.user_scale[user_id](user_attn_energies)
                    attn_energies[i] = user_attn_energies
                attn_weights = F.softmax(attn_energies.squeeze(), dim=1)

            # apply attention weights
            user_representations = torch.bmm(attn_weights.unsqueeze(1), all_session_representations)

            user_representations = self.dropout2(user_representations)

            return user_representations.transpose(0, 1)

        else:
            raise Exception("Invalid method")

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
        #embedded_input = self.dropout1(input_embedding)

        gru_output, hidden = self.gru(input_embedding, hidden)

        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden, input_embedding

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
