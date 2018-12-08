import numpy as np
import string
import re

file = 'embeddings/fr.tsv'
txt = open(file, encoding='utf8').readlines()
vocab_length, vec_dim = [int(i) for i in txt[0].split()]
txt = txt[1:]

# word_idx = 0
embeddings = []
number_list = []
for item in txt:
    item = re.sub('\s+', ' ', item)
    split = item.strip().split(' ')
    split = list(filter(None,split))
    if split[0].isdigit():
        # beginning of a new word
        if len(number_list) != 0:
            embeddings.append(number_list)
        number_list = []
        number_list.append(split[1])
        for each in split[2:]:
            number_list.append(each)
        # word_idx = word_idx + 1
    else:
        for each in split:
            number_list.append(each)

output = open("fr.txt", "a")
output.write(str(vocab_length) + ' ')
output.write(str(vec_dim) + '\n')

for each in embeddings:
    output.write(' '.join([str(x) for x in each]))
    output.write('\n')

output.close()
