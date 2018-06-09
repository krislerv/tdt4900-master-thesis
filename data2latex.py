import numpy as np

# lastfm, reddit
baseline_filename = "baseline-reddit_low_low"
filename = "hierarchical-ATTN_L-ATTN_L-reddit_low_low"

baseline_file = open("data/" + baseline_filename + ".txt")
file = open("data/" + filename + ".txt")

def get_mean_and_std(file):

	input = []

	#shuffle (first all MRR, then all R)
	new_order = [1, 3, 5, 0, 2, 4]

	for line in file:
		line = line.split("\t")
		new_input = []
		for i in new_order:
			new_input.append(float(line[i]))
		input.append(new_input)

	# calculate mean and std
	mean = np.mean(input, axis=0)
	std = np.std(input, axis=0)

	# round to 4 digits
	mean = np.around(mean, decimals=4)
	std = np.around(std, decimals=4)

	return mean, std

b_mean, b_std = get_mean_and_std(baseline_file)
mean, std = get_mean_and_std(file)

mean_percentage_difference = []
for i in range(len(b_mean)):
	mean_percentage_difference.append(((mean[i] / b_mean[i]) - 1) * 100)
mean_percentage_difference = np.around(mean_percentage_difference, decimals=1)

# create final latex string
out = ""

# print baseline result
for i in range(len(mean)):
	if b_std[i] == 0:
		out += " & $" + str(b_mean[i]) + "$"
	else:
		out += " & $" + str(b_mean[i]) + "\pm" + str(b_std[i]) + "$"

out += " \\\\\n"

print(out)

out = ""

# print other result
for i in range(len(mean)):
	if std[i] == 0:
		out += " & $" + str(mean[i]) + "$"
	else:
		out += " & $" + str(mean[i]) + "\pm" + str(std[i]) + "$"

out += " \\\\\n"

# print percentage difference
for i in range(len(mean_percentage_difference)):
	if mean_percentage_difference[i] > 0:
		out += " & (+" + str(mean_percentage_difference[i]) + "\\%)"
	else:
		out += " & (" + str(mean_percentage_difference[i]) + "\\%)"

out += " \\\\"

print(out)