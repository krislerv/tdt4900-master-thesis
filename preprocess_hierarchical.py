import pandas as pd
from itertools import groupby 
from collections import OrderedDict
import json
import os

dataset = "imdb.csv"

HOME = os.path.expanduser('~')
DATASET_PATH = HOME + '/datasets/' + dataset

df = pd.read_csv(DATASET_PATH, sep="\t\t", header=None, dtype={
            "user_id" : str,
            "item_id" : str,
            "rating" : str,
            "review" : str  
        })

tokens = {"<sssss>": 0}
removable_tokens = ["' ", "` ", "´ ", "`` ", "'' ", ", ", ". ", "-lrb- ", "-rrb- "]
results = []

for index, row in df.iterrows():
    review = row[3]
    for t in removable_tokens:
        review.replace(t, "")
    review_tokens = review.split(" ")
    review_tokenized = []
    current_sentence_tokenized = []
    for t in review_tokens:
        if t not in tokens.keys():
            tokens[t] = len(tokens)
        if tokens[t] == 0:
            review_tokenized.append(current_sentence_tokenized)
            current_sentence_tokenized = []
            continue
        current_sentence_tokenized.append(tokens[t])

    results.append(OrderedDict([("user_id", row[0]),
                                ("item_id", row[1]),
                                ("rating", row[2]),
                                ("tokens", review_tokenized)]))

print(len(tokens))
with open(HOME + '/datasets/' + 'imdb_final.json', 'w') as outfile:
    outfile.write(json.dumps(results, indent=4))