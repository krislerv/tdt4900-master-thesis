import os
import pickle
import matplotlib.pyplot as plt


HOME = os.path.expanduser('~')

#################################
#         country count		    #
#################################


"""
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
"""

#################################
#    timestamp distribution		#
#################################
"""
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
"""

#################################
#    average session length		#
#################################

dataset = "lastfm cet"

print("Dataset:", dataset)

dataset_path = dataset + '/4_train_test_split.pickle'

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

kk = 0

for k, v in test_session_lengths.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
    kk += 1
    for session_length in v:
        sum += session_length
        count += 1
        test_sum += session_length
        test_count += 1
print(kk)

print("Train avg session length:", test_sum/test_count)
print("Train session count:", test_count)

print("Total avg session length:", sum/count)
print("Total session count:", count)


#################################
#   session gap distribution	#
#################################


"""
dataset = "lastfm-large"

dataset_path = HOME + '/datasets/' + dataset + '/4_train_test_split.pickle'

dataset = pickle.load(open(dataset_path, 'rb'))

trainset = dataset['trainset']
testset = dataset['testset']
train_session_lengths = dataset['train_session_lengths']
test_session_lengths = dataset['test_session_lengths']

num_users = len(trainset)
if len(trainset) != len(testset):
    raise Exception("Testset and trainset have different amount of users.")

session_gaps = []

for k, v in trainset.items():  # k = user id, v = sessions (list containing lists (sessions) containing lists (tuples of epoch timestamp, event aka artist/subreddit id))
    for session_index in range(1, len(v)):
        gap = (trainset[k][session_index][0][0]-trainset[k][session_index-1][0][0])/3600
        session_gaps.append(gap)

print(len(session_gaps))
print(min(session_gaps))
print(max(session_gaps))

plt.hist(session_gaps, 1000, range=(0, 1000), log=True, color='#0000FF', edgecolor='none')
plt.show()
"""
