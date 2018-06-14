import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


HOME = os.path.expanduser('~')


def country_count():
    dataset = "lastfm"
    dataset_path = HOME + '/datasets/' + dataset + '/userid-profile.tsv'
    dataset = open(dataset_path, 'r')

    cet = [
    "norway",
    "sweden",
    "denmark",
    "germany",
    "france",
    "netherlands",
    "belgium",
    "spain",
    "italy",
    "switzerland",
    "austria",
    "poland",
    "czech republic",
    "slovenia",
    "slovakia",
    "croatia",
    "hungary",
    "bosnia hercegovina"
    ]


    a = 0
    b = 0
    c = 0
    for line in dataset:
    	line = line.split('\t')
    	country = line[3]
    	if country == 'United States':
    		a += 1
    	elif country == "United Kingdom":
    		b += 1
    	if country.lower() in cet:
    		c += 1

    print(a)
    print(b)
    print(c)


def timestamp_distribution():
    dataset = "lastfm orig"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    num_users = len(trainset)
    if len(trainset) != len(testset):
        raise Exception("Testset and trainset have different amount of users.")

    timestamps = []

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_index in range(len(v)):
            timestamp = trainset[k][session_index][0][0]
            timestamps.append(timestamp)
    for k, v in testset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_index in range(len(v)):
            timestamp = trainset[k][session_index][0][0]
            timestamps.append(timestamp)

    print(len(timestamps))
    print(min(timestamps))
    print(max(timestamps))

    plt.hist(timestamps, 1500, color='#0000FF', edgecolor='none')
    plt.show()

def avg_session_length():
    dataset = "subreddit"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    sum = 0
    count = 0

    for k, v in train_session_lengths.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_length in v:
            sum += session_length
            count += 1

    print("Train avg session length:", sum/count)
    print("Train session count:", count)

    test_sum = 0
    test_count = 0

    for k, v in test_session_lengths.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_length in v:
            sum += session_length
            count += 1
            test_sum += session_length
            test_count += 1

    print("Test avg session length:", test_sum/test_count)
    print("Test session count:", test_count)

    print("Total avg session length:", sum/count)
    print("Total session count:", count)

def session_gap_distribution():
    dataset = "lastfm-large"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    session_gaps = []

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_index in range(1, len(v)):
            gap = (trainset[k][session_index][0][0]-trainset[k][session_index-1][0][0])/3600
            session_gaps.append(gap)

    print(len(session_gaps))
    print(min(session_gaps))
    print(max(session_gaps))

    plt.hist(session_gaps, 1000, range=(0, 400), log=True, color='#0000FF', edgecolor='none')
    plt.xticks(np.arange(0, 361, 24))
    plt.show()


def num_unique_actions_per_user():
    dataset = "subreddit"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    per_user_unique_actions = {}
    per_user_total_actions = {}

    for i in range(len(trainset.items())):
        per_user_total_actions[i] = 0

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        user_unique_items = set()
        for session_index in range(len(v)):
            for event_index in range(len(trainset[k][session_index])):
                user_unique_items.add(trainset[k][session_index][0][1])
                per_user_total_actions[k] += 1
        per_user_unique_actions[k] = user_unique_items

    for k, v in per_user_unique_actions.items():
        per_user_unique_actions[k] = len(v)

    ##### TEST SET OGSÅ

    return per_user_unique_actions, per_user_total_actions

def avg_session_length_per_user():
    dataset = "subreddit"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    per_user_session_lengths = {}

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        user_session_lengths = []
        for session_index in range(len(v)):
            event_count = 0
            for event_index in range(len(trainset[k][session_index])):
                if trainset[k][session_index][event_index][1] != 0:
                    event_count += 1
                else:
                    break
            user_session_lengths.append(event_count)
        per_user_session_lengths[k] = user_session_lengths

    for k, v in per_user_session_lengths.items():
        per_user_session_lengths[k] = sum(per_user_session_lengths[k]) / len(per_user_session_lengths[k])

    ##### TEST SET OGSÅ

    return list(per_user_session_lengths.values())

def plot_num_unique_user_actions_vs_accuracy_increase():
    baseline = open("reddit_baseline_per_user_accuracy.txt", "r", encoding="utf-8")
    hidden = open("reddit_hidden_per_user_accuracy.txt", "r", encoding="utf-8")

    per_user_unique_actions, per_user_total_actions = num_unique_actions_per_user()

    line = baseline.readline()
    baseline = line.split(",")

    line = hidden.readline()
    hidden = line.split(",")

    per_user_accuracy_increase = []

    for i in range(len(per_user_unique_actions)):
        per_user_accuracy_increase.append(float(hidden[i].strip()) - float(baseline[i].strip()))

    for k, v in per_user_unique_actions.items():
        per_user_unique_actions[k] /= per_user_total_actions[k]

    plt.scatter(list(per_user_unique_actions.values()), per_user_accuracy_increase)

    #plt.hist(session_gaps, 1000, range=(0, 400), log=True, color='#0000FF', edgecolor='none')
    #plt.xticks(np.arange(0, 361, 24))
    plt.show()

def plot_user_avg_session_lengths_vs_accuracy_increase():
    baseline = open("reddit_baseline_per_user_accuracy.txt", "r", encoding="utf-8")
    hidden = open("reddit_hidden_per_user_accuracy.txt", "r", encoding="utf-8")

    per_user_avg_session_length = avg_session_length_per_user()

    line = baseline.readline()
    baseline = line.split(",")

    line = hidden.readline()
    hidden = line.split(",")

    per_user_accuracy_increase = []

    for i in range(len(per_user_avg_session_length)):
        per_user_accuracy_increase.append(float(hidden[i].strip()) - float(baseline[i].strip()))

    plt.scatter(per_user_avg_session_length, per_user_accuracy_increase)

    #plt.hist(session_gaps, 1000, range=(0, 400), log=True, color='#0000FF', edgecolor='none')
    #plt.xticks(np.arange(0, 361, 24))
    plt.show()



def time_between_first_and_last_session_per_user():
    dataset = "lastfm"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    per_user_unique_actions = {}
    per_user_total_actions = {}

    for i in range(len(trainset.items())):
        per_user_total_actions[i] = 0

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        num_sessions = len(v)
        print(trainset[k][num_sessions - 1])
        delta_time = trainset[k][num_sessions - 1] - trainset[k][0]
        print(delta_time)

def avg_session_count():
    dataset = "subreddit"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    tr_num_users = 0
    num_sessions = 0

    for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        tr_num_users += 1
        num_sessions += len(v)

    te_num_users = 0

    for k, v in testset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        te_num_users += 1
        num_sessions += len(v)

    assert tr_num_users == te_num_users

    print("Average session count: ", num_sessions / tr_num_users)

def users_with_higher_than_average_session_lengths():
    dataset = "lastfm-full"
    dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'
    dataset = pickle.load(open(dataset_path, 'rb'))

    trainset = dataset['trainset']
    testset = dataset['testset']
    train_session_lengths = dataset['train_session_lengths']
    test_session_lengths = dataset['test_session_lengths']

    user_event_counts = [0]*1000
    user_session_counts = [0]*1000

    for k, v in train_session_lengths.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_length in v:
            user_event_counts[k] += session_length
            user_session_counts[k] += 1

    for k, v in test_session_lengths.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
        for session_length in v:
            user_event_counts[k] += session_length
            user_session_counts[k] += 1


    user_avg_session_lengths = []

    for i in range(len(user_event_counts)):
        if user_session_counts[i] == 0:
            continue
        user_avg_session_lengths.append(user_event_counts[i]/user_session_counts[i])

    print(len(user_avg_session_lengths))


    print("Per user average session lengths: ", user_avg_session_lengths)

avg_session_length()
avg_session_count()
