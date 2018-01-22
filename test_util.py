class Tester:

    def __init__(self, num_users, k=[5, 10, 20]):
        self.k = k
        self.session_length = 19
        self.n_decimals = 4
        self.num_users = num_users
        self.initialize()

    def initialize(self):
        self.i_count = [0]*19
        self.recall = [[0]*len(self.k) for i in range(self.session_length)]
        self.mrr = [[0]*len(self.k) for i in range(self.session_length)]

        self.correct_predictions_per_user = [0]*self.num_users
        self.total_predictions_per_user = [0]*self.num_users

        self.correct_predictions_per_session_length = [0]*30
        self.total_predictions_per_session_length = [0]*30

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len, user_id):
        for i in range(seq_len):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k)):
                k = self.k[j]
                if target_item in k_predictions[:k].data:
                    self.recall[i][j] += 1
                    inv_rank = 1.0/self.get_rank(target_item, k_predictions[:k].data)
                    self.mrr[i][j] += inv_rank
                    if k == 20:
                        self.correct_predictions_per_user[user_id] += 1
                        self.correct_predictions_per_session_length[seq_len] += 1
            self.total_predictions_per_user[user_id] += 1
            self.i_count[i] += 1
        self.total_predictions_per_session_length[seq_len] += seq_len


    def evaluate_batch(self, predictions, targets, sequence_lengths, user_list):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            user_id = user_list[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index], user_id)
    
    def format_score_string(self, score_type, score):
        tabs = '\t'
        #if len(score_type) < 8:
        #    tabs += '\t'
        return '\t'+score_type+tabs+score+'\n'

    def get_stats(self):
        score_message = "Recall@5\tMRR@5\tRecall@10\tMRR@10\tRecall@20\tMRR@20\n"
        current_recall = [0]*len(self.k)
        current_mrr = [0]*len(self.k)
        current_count = 0
        recall_k = [0]*len(self.k)
        for i in range(self.session_length):
            score_message += "\ni<="+str(i)+"\t"
            current_count += self.i_count[i]
            for j in range(len(self.k)):
                current_recall[j] += self.recall[i][j]
                current_mrr[j] += self.mrr[i][j]
                k = self.k[j]

                r = current_recall[j]/current_count
                m = current_mrr[j]/current_count
                
                score_message += str(round(r, self.n_decimals))+'\t'
                score_message += str(round(m, self.n_decimals))+'\t'

                recall_k[j] = r

        recall5 = recall_k[0]
        recall20 = recall_k[2]

        per_user_accuracies = []
        for i in range(self.num_users):
            if self.total_predictions_per_user[i] == 0:
                per_user_accuracies.append(0)
            else:
                per_user_accuracies.append(self.correct_predictions_per_user[i] / self.total_predictions_per_user[i])

        per_session_length_accuracies = []
        for i in range(self.session_length):
            if self.total_predictions_per_session_length[i] == 0:
                per_session_length_accuracies.append(0)
            else:
                per_session_length_accuracies.append(self.correct_predictions_per_session_length[i] / self.total_predictions_per_session_length[i])

        return score_message, recall5, recall20, per_user_accuracies, per_session_length_accuracies

    def get_stats_and_reset(self):
        message = self.get_stats()
        self.initialize()
        return message
