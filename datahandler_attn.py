import collections
import datetime
import logging
import math
import numpy as np
import os
import pickle
import time

class IIRNNDataHandler:
    
    def __init__(self, dataset_path, batch_size, log_file, max_sess_reps, lt_internalsize):
        # LOAD DATASET
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        print("Loading dataset")
        load_time = time.time()
        dataset = pickle.load(open(self.dataset_path, 'rb'))
        print("|- dataset loaded in", str(time.time()-load_time), "s")

        self.trainset = dataset['trainset']
        self.testset = dataset['testset']
        self.train_session_lengths = dataset['train_session_lengths']
        self.test_session_lengths = dataset['test_session_lengths']
    
        self.num_users = len(self.trainset)
        if len(self.trainset) != len(self.testset):
            raise Exception("""Testset and trainset have different 
                    amount of users.""")

        # II_RNN stuff
        self.MAX_SESSION_REPRESENTATIONS = max_sess_reps
        self.LT_INTERNALSIZE = lt_internalsize

        # LOG
        self.log_file = log_file
        logging.basicConfig(filename=log_file, level=logging.DEBUG)

        self.set_session_timestamps()
    
        # batch control
        self.reset_user_batch_data()

    def set_session_timestamps(self):
        self.train_timestamps = [None] * self.num_users
        self.test_timestamps = [None] * self.num_users
        self.train_timestamp_bucket_ids = [None] * self.num_users
        self.test_timestamp_bucket_ids = [None] * self.num_users
        for k, v in self.trainset.items():
            timestamps = []
            timestamp_bucket_ids = []
            for session_index in range(len(v)):
                timestamp = self.trainset[k][session_index][0][0]
                timestamps.append(timestamp)
                timestamp_bucket_ids.append(datetime.datetime.utcfromtimestamp(timestamp).weekday() * 24 + datetime.datetime.utcfromtimestamp(timestamp).hour)
            self.train_timestamps[k] = timestamps
            self.train_timestamp_bucket_ids[k] = timestamp_bucket_ids
        for k, v in self.testset.items():
            timestamps = []
            timestamp_bucket_ids = []
            for session_index in range(len(v)):
                timestamp = self.testset[k][session_index][0][0]
                timestamps.append(timestamp)
                timestamp_bucket_ids.append(datetime.datetime.utcfromtimestamp(timestamp).weekday() * 24 + datetime.datetime.utcfromtimestamp(timestamp).hour)
            self.test_timestamps[k] = timestamps
            self.test_timestamp_bucket_ids[k] = timestamp_bucket_ids

    """
    Returns the unix time for a given event in the last session processed for the given user
    event_id: The id of the event in the last session
    user_id: The user to retrieve event for
    """
    def get_event_time_of_last_session_for_given_user(self, event_id, user_id):
        return self.trainset[user_id][self.user_next_session_to_retrieve[user_id] - 1][event_id][0]

    def get_last_sessions_for_user(self, user_id):
        last_sessions = []
        session_index = self.user_next_session_to_retrieve[user_id] - 2 # minus two because minus one is currently being used as xinput
        for i in range(session_index, session_index - 15, -1):
            if i < 0:
                break
            last_sessions.append(self.trainset[user_id][i])
        return last_sessions


    # call before training and testing
    def reset_user_batch_data(self):
        # the index of the next session(event) to retrieve for a user
        self.user_next_session_to_retrieve = [0]*self.num_users
        # list of users who have not been exhausted for sessions
        self.users_with_remaining_sessions = []
        # a list where we store the number of remaining sessions for each user. Updated for eatch batch fetch. But we don't want to create the object multiple times.
        self.num_remaining_sessions_for_user = [0]*self.num_users
        for k, v in self.trainset.items():
            # everyone has at least one session
            self.users_with_remaining_sessions.append(k)

    def reset_user_session_representations(self):
        # session representations for each user is stored here
        self.user_session_representations = [None]*self.num_users
        self.user_session_representations_timestamps = [None]*self.num_users
        self.user_session_representations_timestamp_bucket_ids = [None]*self.num_users
        # the number of (real) session representations a user has
        self.num_user_session_representations = [0]*self.num_users
        for k, v in self.trainset.items():
            self.user_session_representations[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            self.user_session_representations[k].append([0]*self.LT_INTERNALSIZE)
            self.user_session_representations_timestamps[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            self.user_session_representations_timestamps[k].append(0)
            self.user_session_representations_timestamp_bucket_ids[k] = collections.deque(maxlen=self.MAX_SESSION_REPRESENTATIONS)
            self.user_session_representations_timestamp_bucket_ids[k].append(0)

    def get_N_highest_indexes(a,N):
        return np.argsort(a)[::-1][:N]

    def add_unique_items_to_dict(self, items, dataset):
        for k, v in dataset.items():
            for session in v:
                for event in session:
                    item = event[1]
                    if item not in items:
                        items[item] = True
        return items

    def get_num_items(self):
        items = {}
        items = self.add_unique_items_to_dict(items, self.trainset)
        items = self.add_unique_items_to_dict(items, self.testset)
        return len(items)

    def get_num_sessions(self, dataset):
        session_count = 0
        for k, v in dataset.items():
            session_count += len(v)
        return session_count

    def get_num_training_sessions(self):
        return self.get_num_sessions(self.trainset)
    
    # for the II-RNN this is only an estimate
    def get_num_batches(self, dataset):
        num_sessions = self.get_num_sessions(dataset)
        return math.ceil(num_sessions/self.batch_size)

    def get_num_training_batches(self):
        return self.get_num_batches(self.trainset)

    def get_num_test_batches(self):
        return self.get_num_batches(self.testset)

    def get_next_batch(self, dataset, dataset_session_lengths, timestamp_set, timestamp_bucket_ids_set):
        session_batch = []
        session_lengths = []
        sess_rep_batch = []
        sess_rep_lengths = []
        sess_time_vectors = []
        sess_rep_timestamps_batch = []
        sess_rep_timestamp_bucket_ids_batch = []
        input_timestamps = []
        input_timestamp_bucket_ids = []
        
        # Decide which users to take sessions from. First count the number of remaining sessions
        remaining_sessions = [0]*len(self.users_with_remaining_sessions)
        for i in range(len(self.users_with_remaining_sessions)):
            user = self.users_with_remaining_sessions[i]
            remaining_sessions[i] = len(dataset[user]) - self.user_next_session_to_retrieve[user]
        
        # index of users to get
        user_list = IIRNNDataHandler.get_N_highest_indexes(remaining_sessions, self.batch_size)
        if(len(user_list) == 0):
            return [], [], [], [], [], [], [], [], [], []
        for i in range(len(user_list)):
            user_list[i] = self.users_with_remaining_sessions[user_list[i]]

        # For each user -> get the next session, and check if we should remove 
        # him from the list of users with remaining sessions
        for user in user_list:
            session_index = self.user_next_session_to_retrieve[user]
            session_batch.append(dataset[user][session_index])
            session_lengths.append(dataset_session_lengths[user][session_index])
            srl = max(self.num_user_session_representations[user],1)
            sess_rep_lengths.append(srl)
            sess_rep = list(self.user_session_representations[user]) #copy
            sess_rep_timestamps = list(self.user_session_representations_timestamps[user])
            sess_rep_timestamp_bucket_ids = list(self.user_session_representations_timestamp_bucket_ids[user])
            if(srl < self.MAX_SESSION_REPRESENTATIONS):
                for i in range(self.MAX_SESSION_REPRESENTATIONS-srl):
                    sess_rep.append([0]*self.LT_INTERNALSIZE) #pad with zeroes after valid reps
                    sess_rep_timestamps.append(0)
                    sess_rep_timestamp_bucket_ids.append(168)
            sess_rep_batch.append(sess_rep)
            sess_rep_timestamps_batch.append(sess_rep_timestamps)
            sess_rep_timestamp_bucket_ids_batch.append(sess_rep_timestamp_bucket_ids)

            self.user_next_session_to_retrieve[user] += 1
            if self.user_next_session_to_retrieve[user] >= len(dataset[user]):
                # User have no more session, remove him from users_with_remaining_sessions
                self.users_with_remaining_sessions.remove(user)

            input_timestamps.append(timestamp_set[user][session_index])
            input_timestamp_bucket_ids.append(timestamp_bucket_ids_set[user][session_index])

        #sort batch based on seq rep len
        session_batch = [[event[1] for event in session] for session in session_batch]
        x = [session[:-1] for session in session_batch]
        y = [session[1:] for session in session_batch]

        return x, y, session_lengths, input_timestamps, input_timestamp_bucket_ids, sess_rep_batch, sess_rep_lengths, sess_rep_timestamps_batch, sess_rep_timestamp_bucket_ids_batch, user_list

    def get_next_train_batch(self):
        return self.get_next_batch(self.trainset, self.train_session_lengths, self.train_timestamps, self.train_timestamp_bucket_ids)

    def get_next_test_batch(self):
        return self.get_next_batch(self.testset, self.test_session_lengths, self.test_timestamps, self.test_timestamp_bucket_ids)

    def get_latest_epoch(self, epoch_file):
        if not os.path.isfile(epoch_file):
            return 0
        return pickle.load(open(epoch_file, 'rb'))
    
    def store_current_epoch(self, epoch, epoch_file):
        pickle.dump(epoch, open(epoch_file, 'wb'))

    
    def add_timestamp_to_message(self, message):
        timestamp = str(datetime.datetime.now())
        message = timestamp+'\n'+message
        return message

    def log_test_stats(self, epoch_number, epoch_loss, stats):
        timestamp = str(datetime.datetime.now())
        message = timestamp+'\n\tEpoch #: '+str(epoch_number)
        message += '\n\tEpoch loss: '+str(epoch_loss)+'\n'
        message += stats
        logging.info(message)

    def log_config(self, config):
        config = self.add_timestamp_to_message(config)
        logging.info(config)

    
    def store_user_session_representations(self, sessions_representations, user_list, session_timestamps, session_timestamp_bucket_ids):
        for i in range(len(user_list)):
            user = user_list[i]
            if(user != -1):
                session_representation = list(sessions_representations[i])
                session_timestamp = float(session_timestamps[i])
                session_timestamp_bucket_id = int(session_timestamp_bucket_ids[i])

                num_reps = self.num_user_session_representations[user]
                

                #self.num_user_session_representations[user] = min(self.MAX_SESSION_REPRESENTATIONS, num_reps+1)
                if(num_reps == 0):
                    self.user_session_representations[user].pop() #pop the sucker
                    self.user_session_representations_timestamps[user].pop()
                    self.user_session_representations_timestamp_bucket_ids[user].pop()
                self.user_session_representations[user].append(session_representation)
                self.user_session_representations_timestamps[user].append(session_timestamp)
                self.user_session_representations_timestamp_bucket_ids[user].append(session_timestamp_bucket_id)
                self.num_user_session_representations[user] = min(self.MAX_SESSION_REPRESENTATIONS, num_reps+1)

    def log_attention_weights_inter(self, use_hidden_state_attn, use_delta_t_attn, use_week_time_attn, user_id, inter_attn_weights, input_timestamps):
        file = open("inter_attn_weights-" + str(use_hidden_state_attn) + '-' + str(use_delta_t_attn) + '-' + str(use_week_time_attn) + ".txt", "a", encoding="utf-8")

        last_sessions_for_user = self.get_last_sessions_for_user(user_id)
        for i in range(15):
            file.write(str(inter_attn_weights[0][i].data[0]) + ",")
        file.write("\n\n")
        file.write(str(input_timestamps[0]))
        file.write("\n\n")
        for session_id in range(len(last_sessions_for_user)):
            file.write(str(last_sessions_for_user[session_id][0][0]) + "\n")
            for event_id in range(len(last_sessions_for_user[session_id])):
                file.write(str(last_sessions_for_user[session_id][event_id][1]) + ",")
            file.write("\n")
        file.write("\n\n\n\n\n\n")
        file.close()

    def log_attention_weights_intra(self, intra_attn_weights, use_hidden_state_attn, use_delta_t_attn, use_week_time_attn, sl, top_k_predictions, user_id):
        intra_attn_weights = intra_attn_weights.transpose(1, 2)
        file = open("intra_attn_weights-" + str(use_hidden_state_attn) + '-' + str(use_delta_t_attn) + '-' + str(use_week_time_attn) + ".txt", "a", encoding="utf-8")
        session_length = sl[0]
        file.write(str(session_length) + "\n")
        for i in range(19):
            file.write(str(top_k_predictions[0][i][0].data[0]) + ",")
        file.write("\n\n")
        last_sessions_for_user = self.get_last_sessions_for_user(user_id)
        for session_id in range(len(last_sessions_for_user)):
            for a in range(len(intra_attn_weights[0][session_id])):
                file.write(str(intra_attn_weights[0][session_id][a].data[0]).format("%0.4f") + ",")
            file.write("\n")
            for event_id in range(len(last_sessions_for_user[session_id])):
                file.write(str(last_sessions_for_user[session_id][event_id][1]) + ",")
            file.write("\n\n")
        file.write("\n\n\n\n\n\n")
        file.close()
