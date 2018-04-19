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
    def __init__(self, hidden_size, dropout, gpu_no=0):
        super(OnTheFlySessionRepresentations, self).__init__()

        self.hidden_size = hidden_size
        self.gpu_no = gpu_no
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_previous_session_batch_embedding, user_previous_session_lengths, user_prevoius_session_counts):
        user_previous_session_batch_embedding_summed = user_previous_session_batch_embedding.sum(1)
        mean_user_previous_session_batch_embedding = user_previous_session_batch_embedding_summed.transpose(0, 1).div(user_previous_session_lengths.float()).transpose(0, 1)

        return mean_user_previous_session_batch_embedding
        


class SesssionRepresentationCreator(nn.Module):
    def __init__(self, hidden_size, dropout, gpu_no=0):
        super(SesssionRepresentationCreator, self).__init__()

        self.hidden_size = hidden_size
        self.gpu_no = gpu_no
        self.dropout = nn.Dropout(dropout)
        self.previous_session_representations = [0] * 1000

    def forward(self, user_list, input_embedding, session_lengths):
        #input_embedding = self.dropout(input_embedding)
        sum_x = input_embedding.sum(1)
        mean_x = sum_x.div(session_lengths.float())

        all_session_representations = Variable(torch.zeros(len(user_list), 15, self.hidden_size)).cuda(self.gpu_no)
        for i in range(len(user_list)):
            user_id = user_list[i].data[0]
            if self.previous_session_representations[user_id] == 0:
                all_session_representations[i] = torch.zeros(15, 100).float()
            elif len(self.previous_session_representations[user_id]) < 15:
                temp_sess_rep = list(self.previous_session_representations[user_id])
                while len(temp_sess_rep) < 15:
                    temp_sess_rep.append([0] * 100)
                all_session_representations[i] = torch.FloatTensor(temp_sess_rep)
            else:
                all_session_representations[i] = torch.FloatTensor(self.previous_session_representations[user_id][-15:])

        for i in range(len(user_list)):
            user_id = user_list[i].data[0]
            if self.previous_session_representations[user_id] == 0:
                self.previous_session_representations[user_id] = [mean_x[i].data.tolist()]
            else:
                self.previous_session_representations[user_id].append(mean_x[i].data.tolist())

        return all_session_representations, mean_x

    def reset_session_representations(self):
        self.previous_session_representations = [0] * 1000

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
