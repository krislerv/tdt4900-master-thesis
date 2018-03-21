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
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, use_hidden_state_attn=False, use_delta_t_attn=False, use_week_time_attn=False, gpu_no=0):
        super(InterRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale = nn.Linear(self.hidden_size, 1, bias=False)

        self.index_list = []
        for i in range(20):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)


    def forward(self, hidden, current_session_batch, current_session_lengths):
        # gets the output of the last non-zero session representation
        current_session_batch = self.dropout1(current_session_batch)
        output, hidden = self.gru(current_session_batch, hidden)
        # create a mask so that attention weights for "empty" outputs are zero
        current_session_lengths_expanded = current_session_lengths.unsqueeze(1).expand(output.size(0), output.size(1))     # [BATCH_SIZE, MAX_SEQ_LEN]
        indexes = self.index_list.unsqueeze(0).expand(output.size(0), output.size(1))                                      # [BATCH_SIZE, MAX_SEQ_LEN]
        mask = torch.lt(indexes, current_session_lengths_expanded) # [BATCH_SIZE, MAX_SEQ_LEN]  i-th element in each batch is 1 if it is a real item, 0 otherwise
        print(mask[0])
        mask = mask.unsqueeze(2).expand(output.size(0), output.size(1), output.size(2)).float() * 1000000 - 1000000   # 1 -> 0, 0 -> -1000000    [BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE]

        output = output + mask  # apply mask

        attn_energies = torch.tanh(self.attention(output))
        attn_energies = self.scale(attn_energies)

        attn_weights = F.softmax(attn_energies.squeeze())

        print(attn_weights[0])

        session_representations = torch.bmm(attn_weights.unsqueeze(1), current_session_batch)   # session representations are a weigted sum of each item embedding of the session, weights determined by attention mechanism

        # gets the last actual hidden state (generated from a non-zero input)
        zeros = Variable(torch.zeros(current_session_lengths.size(0)).long()).cuda()
        last_index_of_sessions = current_session_lengths - 1
        last_index_of_sessions = torch.max(zeros, last_index_of_sessions) # if last index is < 0, set it to 0
        hidden_indices = last_index_of_sessions.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
        hidden = torch.gather(output, 1, hidden_indices)
        hidden = hidden.squeeze()
        hidden = hidden.unsqueeze(0)
        #hidden = self.dropout2(hidden)

        session_representations = self.dropout2(session_representations)

        return output, hidden, attn_weights, session_representations

    # initialize hidden with variable batch size
    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden


class InterRNN2(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, use_hidden_state_attn=False, use_delta_t_attn=False, use_week_time_attn=False, gpu_no=0):
        super(InterRNN2, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.scale = nn.Linear(self.hidden_size, 1, bias=False)

        self.index_list = []
        for i in range(self.max_session_representations):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)


    def forward(self, hidden, all_session_representations, prevoius_session_counts):
        all_session_representations = self.dropout1(all_session_representations)
        output, hidden = self.gru(all_session_representations, hidden)

        # create a mask so that attention weights for "empty" outputs are zero
        prevoius_session_counts_expanded = prevoius_session_counts.unsqueeze(1).expand(output.size(0), output.size(1))     # [BATCH_SIZE, MAX_SESS_REP]
        indexes = self.index_list.unsqueeze(0).expand(output.size(0), output.size(1))                                      # [BATCH_SIZE, MAX_SESS_REP]
        mask = torch.lt(indexes, prevoius_session_counts_expanded) # [BATCH_SIZE, MAX_SESS_REP]  i-th element in each batch is 1 if it is a real sess rep, 0 otherwise
        mask = mask.unsqueeze(2).expand(output.size(0), output.size(1), output.size(2)).float() * 1000000 - 1000000   # 1 -> 0, 0 -> -1000000    [BATCH_SIZE, MAX_SESS_REP, HIDDEN_SIZE]

        output = output + mask  # apply mask

        attn_energies = torch.tanh(self.attention(output))
        attn_energies = self.scale(attn_energies)

        attn_weights = F.softmax(attn_energies.squeeze())

        user_representations = torch.bmm(attn_weights.unsqueeze(1), output)

        # gets the last actual hidden state (generated from a non-zero input)
        zeros = Variable(torch.zeros(prevoius_session_counts.size(0)).long()).cuda()
        last_index_of_sessions = prevoius_session_counts - 1
        last_index_of_sessions = torch.max(zeros, last_index_of_sessions) # if last index is < 0, set it to 0
        hidden_indices = last_index_of_sessions.view(-1, 1, 1).expand(output.size(0), 1, output.size(2))
        hidden = torch.gather(output, 1, hidden_indices)
        hidden = hidden.squeeze()
        hidden = hidden.unsqueeze(0)
        hidden = self.dropout2(hidden)

        return output, hidden, attn_weights, user_representations

    # initialize hidden with variable batch size
    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden

class IntraRNN(nn.Module):
    def __init__(self, n_items, hidden_size, embedding_size, n_layers, dropout, max_session_representations, use_attn=False, use_delta_t_attn=False, use_per_user_intra_attn=False, gpu_no=0):
        super(IntraRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.use_attn = use_attn
        self.use_delta_t_attn = use_delta_t_attn
        self.use_per_user_intra_attn = use_per_user_intra_attn

        #self.embedding = nn.Embedding(n_items, embedding_size)
        #self.embedding.weight.data.copy_(torch.zeros(n_items, embedding_size).uniform_(-1, 1))
        #self.embedding.weight.data[0] = torch.zeros(embedding_size) # ensure that the representation of paddings are tensors of zeros, which then easily can be used in an average rep
        self.gru = nn.GRU(2 * embedding_size if use_attn else embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, n_items)

        if self.use_attn:
            #self.wa = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
            #self.wa.data.copy_(torch.zeros(hidden_size, hidden_size).uniform_(-1, 1))
            #self.ua = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
            #self.ua.data.copy_(torch.zeros(hidden_size, hidden_size).uniform_(-1, 1))
            #self.va = nn.Parameter(torch.FloatTensor(1, 2 * hidden_size))
            #self.va.data.copy_(torch.zeros(1, 2 * hidden_size).uniform_(-1, 1))
            #self.fff = nn.Parameter(torch.FloatTensor((3 if use_delta_t_attn else 2) * self.hidden_size, 2 * self.hidden_size))
            #self.fff.data.copy_(torch.zeros((3 if use_delta_t_attn else 2) * self.hidden_size, 2 * self.hidden_size).uniform_(-1, 1))

            #self.va = nn.Linear(2 * self.hidden_size, 1, bias=False)
            #self.fff = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False)
            self.wa = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.ua = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.va = nn.Linear(self.hidden_size, 1, bias=False)
            self.fff = nn.Linear((3 if use_delta_t_attn else 2) * self.hidden_size, 1, bias=False)

            self.lin1 = nn.Linear(hidden_size, 1)
            self.lin2 = nn.Linear(hidden_size, 1)

            self.hidden_linear = nn.Linear(hidden_size, hidden_size)
            self.inter_linear = nn.Linear(hidden_size, hidden_size)

            self.delta_embedding = nn.Embedding(169, embedding_size)
            self.delta_embedding.weight.data.copy_(torch.zeros(169, embedding_size).uniform_(-1, 1))

            self.user_embedding = nn.Embedding(1000, embedding_size) # TODO: don't hardcode num_users
            self.user_embedding.weight.data.copy_(torch.zeros(1000, embedding_size).uniform_(-1, 1))

            if use_per_user_intra_attn:
                self.inter_params = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1000)])
                self.hidden_params = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1000)])
                self.scale_params = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(1000)])
                self.linear_test1_params = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size) for i in range(1000)])

            self.linear_test1 = nn.Linear(2*hidden_size, hidden_size)
            self.linear_test2 = nn.Linear(2*hidden_size, hidden_size)


    def forward(self, input, input_embedding, hidden, inter_output, user_list):
        #embedded_input = self.embedding(input)
        embedded_input = self.dropout1(input_embedding)

        if self.use_attn:
            ### ALL
            #user_list = self.user_embedding(user_list)
            #delta_t_h = self.delta_embedding(delta_t_h)
            hidden_t = hidden.transpose(0, 1)

            ### BASELINE
            if self.use_delta_t_attn:
                cat = torch.cat((hidden_t.expand(input.size(0), self.max_session_representations, self.hidden_size), inter_output, delta_t_h), dim=2)
            else:
                if self.use_per_user_intra_attn:
                    inter_output_a = Variable(torch.zeros(input.size(0), self.max_session_representations, self.hidden_size)).cuda(self.gpu_no)
                    hidden_t_a = Variable(torch.zeros(input.size(0), 1, self.hidden_size)).cuda(self.gpu_no)
                    result = Variable(torch.zeros(input.size(0), self.max_session_representations, self.hidden_size)).cuda(self.gpu_no)
                    for i in range(input.size(0)):
                        inter_output_a[i] = self.inter_params[user_list[i].data[0]](inter_output[i])
                        hidden_t_a[i] = self.hidden_params[user_list[i].data[0]](hidden_t[i])
                        result[i] = torch.tanh(self.linear_test1_params[user_list[i].data[0]](torch.cat((hidden_t_a[i].expand(self.max_session_representations, self.hidden_size), inter_output_a[i]), dim=1)))
                else:
                    hidden_t_a = self.wa(hidden_t)
                    inter_output_a = self.ua(inter_output)
                    #hidden_t_a = hidden_t
                    #inter_output_a = inter_output
                    result = torch.tanh(self.linear_test1(torch.cat((hidden_t_a.expand(input.size(0), self.max_session_representations, self.hidden_size), inter_output_a), dim=2)))  # concat last hidden and inter output
                    #result = torch.tanh(hidden_t_a.expand(input.size(0), self.max_session_representations, self.hidden_size) + inter_output_a)                      # sum last hidden and inter output
            if self.use_per_user_intra_attn:
                energies = Variable(torch.zeros(input.size(0), self.max_session_representations, 1)).cuda(self.gpu_no)
                for i in range(input.size(0)):
                    energies[i] = self.scale_params[user_list[i].data[0]](result[i])
            else:
                energies = self.va(result)
            attn_weights = F.softmax(energies.squeeze(), dim=1)

            #result = torch.tanh(self.fff(cat))
            #attn_weights = F.softmax(result.squeeze(), dim=1)

            # finds the top k attention weights and sets all other attention weights to zero
            #k = 3
            #attn_weights = torch.ge(attn_weights, torch.gather(attn_weights, 1, torch.index_select(torch.topk(attn_weights, k)[1], 1, Variable(torch.LongTensor([k-1])).cuda(self.gpu_no)))).float() * attn_weights



            ### scale, sum, tanh
            #a = self.lin1(hidden_t)
            #b = self.lin2(inter_output)
            #c = a + b
            #result = torch.tanh(c)


            ### cat, scale, tanh
            #a = torch.cat((hidden_t.expand(input.size(0), self.max_session_representations, self.hidden_size), inter_output, delta_t_h), dim=2)
            #c = self.lin1(a)
            #result = torch.tanh(c)


            ### sum, scale, tanh (with extra learnable weights)
            #hidden_t_1 = self.hidden_linear(hidden_t)
            #inter_output_1 = self.inter_linear(inter_output)
            #a = hidden_t_1 + inter_output_1 # + delta_t_h
            #c = self.lin1(a)
            #result = torch.tanh(c)


            
            ### ALL NON-BASELINE
            #result_t = result.transpose(1, 2)
            #attn_weights = F.softmax(result_t.squeeze(), dim=1)


            ### ALL
            context = torch.bmm(attn_weights.unsqueeze(1), inter_output)
            gru_input = torch.cat((embedded_input, context), 2)
            gru_output, hidden = self.gru(gru_input, hidden)
        else:
            gru_output, hidden = self.gru(embedded_input, hidden)
            attn_weights = []

        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden, embedded_input, gru_output, attn_weights

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
