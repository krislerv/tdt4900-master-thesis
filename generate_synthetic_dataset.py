import random
import datetime

genres = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
public_artist_genres = []
private_artist_genres = []

lastfm_session_time_delta = 60*30 # minutes to split sessions
num_artists = 40000
num_sessions = 70000
session_length = 12
num_users = 1000 # doesn't really matter because all users behave the same, but for the sake of mini-batching this is better
sessions_per_user = num_sessions / num_users
num_genres = len(genres)
prob_public_private_equal = 0.7

print("Generating public and private genres")

for i in range(num_artists):
	if random.random() < prob_public_private_equal: # public and private the same
		public_artist_genres.append(genres[i % num_genres])
		private_artist_genres.append(genres[i % num_genres])
	else:	# public and private different (most of the time)
		public_artist_genres.append(genres[i % num_genres])
		private_artist_genres.append(genres[random.randint(0, num_genres - 1)])

print("Filling dictionary")

public_artists = {}
for genre in genres:
	public_artists[genre] = []


for i in range(num_artists):	# add each artist index to the correct genre dictionary  {'A': [1, 2, 3], 'B': [4, 5, 6]} etc etc
	public_artists[public_artist_genres[i]].append(i)


print("Generating sessions")

### generate sessions
sessions = []

for i in range(num_sessions):
	# first event generated randomly
	new_session = [random.randint(0, num_artists - 1)]
	for i in range(session_length - 1):
		last_artist = new_session[-1]
		last_artist_public_genre = public_artist_genres[last_artist]
		last_artist_private_genre = private_artist_genres[last_artist]
		if last_artist_public_genre == last_artist_private_genre:
			candidate_artists = public_artists[last_artist_public_genre]
			new_artist = candidate_artists[random.randint(0, len(candidate_artists) - 1)]
			while new_artist == last_artist:	# in case we randomly picked the same one as before
				new_artist = candidate_artists[random.randint(0, len(candidate_artists) - 1)]
			new_session.append(new_artist)
		else:
			candidate_artists = public_artists[last_artist_private_genre]
			new_artist = candidate_artists[random.randint(0, len(candidate_artists) - 1)]
			new_session.append(new_artist)	# don't need to check if we picked the same artist because the last artist was from a different genre
	sessions.append(new_session)

print("Writing to file")

file = open('synthetic-dataset.tsv', 'w')

# each event needs a timestamp, their value doesn't matter, just make sure they're in the right sequence
time_now = datetime.datetime.now()
timestamps_now = time_now.timestamp()
timestamp_counter = 0
user_session_counter = 0
current_user = 1

for session in sessions:
	user_session_counter += 1
	timestamp_counter += lastfm_session_time_delta * 2 # so that sessions are split correctly
	for artist in session:
		# create timestamp
		timestamp = datetime.datetime.fromtimestamp(timestamps_now + timestamp_counter).strftime('%Y-%m-%dT%H:%M:%SZ')
		file.write(str(current_user) + "\t" + timestamp + "\t" + str(artist) + "\t" + str(artist) + "\n")
		timestamp_counter += 1
	if user_session_counter == sessions_per_user:
		current_user += 1
		user_session_counter = 0

