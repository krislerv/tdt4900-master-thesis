from tensorboard import Logger as TensorBoard

tensorboard = TensorBoard('./logs')

file = open("testlog manual/2018-05-22-15-21-18-testing-attn-rnn-subreddit.txt", "r")

epoch = 1

for line in file:
	if line.startswith("i<=18"):
		line = [x for x in line.split("\t")[1:] if x != '']
		print(line)

		tensorboard.scalar_summary('Recall@5', float(line[0].strip()), epoch)
		tensorboard.scalar_summary('Recall@10', float(line[2].strip()), epoch)
		tensorboard.scalar_summary('Recall@20', float(line[4].strip()), epoch)
		tensorboard.scalar_summary('MRR@5', float(line[1].strip()), epoch)
		tensorboard.scalar_summary('MRR@10', float(line[3].strip()), epoch)
		tensorboard.scalar_summary('MRR@20', float(line[5].strip()), epoch)

		epoch += 1
