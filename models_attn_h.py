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

        self.dropout = nn.Dropout(dropout)

        self.index_list = []
        for i in range(20):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)

        self.previous_session_representations = [0] * 1000


    def forward(self, user_list, input_embedding, session_lengths):
        """
        current_session_batch = self.dropout(current_session_batch) # in the baseline-baseline model, the embeddings are passed through a dropout layer in IntraRNN before being used to create session representations
        prevoius_session_length_is_zero = torch.lt(current_session_lengths, 1)  # padding sessions have length zero (real sessions have minimum length 2)
        current_session_lengths = current_session_lengths + prevoius_session_length_is_zero.long()  # add one to session length if session count is zero (to avoid division by zero error)

        indexes = self.index_list.unsqueeze(0).expand(current_session_batch.size(0), current_session_batch.size(1))
        current_session_lengths_expanded = current_session_lengths.unsqueeze(1).expand(current_session_batch.size(0), current_session_batch.size(1))
        mask = torch.lt(indexes, current_session_lengths_expanded)
        mask = mask.unsqueeze(2).expand(current_session_batch.size(0), current_session_batch.size(1), current_session_batch.size(2)).float()

        current_session_batch = current_session_batch * mask

        sum_x = current_session_batch.sum(1)
        mean_x = sum_x.div(current_session_lengths.unsqueeze(1).float())
        return mean_x
        """
        input_embedding = self.dropout(input_embedding)
        sum_x = input_embedding.sum(1)
        mean_x = sum_x.div(session_lengths.float())

        all_session_representations = Variable(torch.zeros(len(user_list), 15, self.hidden_size)).cuda(self.gpu_no)
        for i in range(len(user_list)):
            if self.previous_session_representations[user_list[i]] == 0:
                all_session_representations[i] = torch.zeros(15, 100).float()
            elif len(self.previous_session_representations[user_list[i]]) < 15:
                temp_sess_rep = list(self.previous_session_representations[user_list[i]])
                while len(temp_sess_rep) < 15:
                    temp_sess_rep.append([0] * 100)
                all_session_representations[i] = torch.FloatTensor(temp_sess_rep)
            else:
                all_session_representations[i] = torch.FloatTensor(self.previous_session_representations[user_list[i]][-15:])

        for i in range(len(user_list)):
            if self.previous_session_representations[user_list[i]] == 0:
                self.previous_session_representations[user_list[i]] = [mean_x[i].data.tolist()]
            else:
                self.previous_session_representations[user_list[i]].append(mean_x[i].data.tolist())

        return all_session_representations, mean_x



    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden

    def reset_session_representations(self):
        self.previous_session_representations = [0] * 1000


class InterRNN2(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers, dropout, max_session_representations, gpu_no=0):
        super(InterRNN2, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_session_representations = max_session_representations

        self.gpu_no = gpu_no

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.index_list = []
        for i in range(self.max_session_representations):
            self.index_list.append(i)
        self.index_list = Variable(torch.LongTensor(self.index_list)).cuda(self.gpu_no)


    def forward(self, hidden, all_session_representations, prevoius_session_counts):
        d_all_session_representations = self.dropout1(all_session_representations)
        output, hidden = self.gru(d_all_session_representations, hidden)

        # find the last non-padded hidden state
        zeros = Variable(torch.zeros(prevoius_session_counts.size(0)).long()).cuda(self.gpu_no)
        last_index_of_session_reps = prevoius_session_counts - 1
        last_index_of_session_reps = torch.max(zeros, last_index_of_session_reps) # if last index is < 0, set it to 0
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
        input_embedding = self.dropout1(input_embedding)

        gru_output, hidden = self.gru(input_embedding, hidden)

        output = self.dropout2(gru_output)
        output = self.linear(output)
        return output, hidden

    def init_hidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(self.gpu_no)
        return hidden
