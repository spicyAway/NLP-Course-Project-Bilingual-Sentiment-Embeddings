import copy
from googletrans import Translator
import os
from nltk.tokenize import sent_tokenize
import random, sys

dataset_type = ['train', 'dev', 'test']
sentiment_type = ['neg', 'pos', 'strneg', 'strpos']
merged_corpus = os.path.join('datasets', 'trans', 'fr', 'opener_sents', 'merged_train_dev_labelled_file.txt')
merged_test_corpus = os.path.join('datasets', 'trans', 'fr', 'opener_sents', 'merged_test_labelled_file.txt')

with open(merged_test_corpus, "a") as merged_test_file:
    with open(merged_corpus, "a") as merged_train_dev_file:
        for type in dataset_type:
            for sentiment in sentiment_type:

                output_file = os.path.join('datasets', 'trans', 'fr', 'opener_sents', type, 'labelled_' + sentiment + '.txt')
                text_file = open(os.path.join('datasets', 'trans', 'fr', 'opener_sents', type, sentiment + '.txt'), "r")
                lines = text_file.read().split('\n')
                print(len(lines))

                count = 0
                total = len(lines)
                with open(output_file, "a") as text_file:
                    for line in lines:
                        paragraph = []
                        sent_tokenize_list = sent_tokenize(line, language='french')
                        label = '__label__' + sentiment
                        paragraph.append(label)
                        for sent in sent_tokenize_list:
                            paragraph.append(sent)
                        temp = ' '.join([str(x) for x in paragraph])
                        text_file.write(temp)
                        text_file.write('\n')

                        if type != 'test':
                            merged_train_dev_file.write(temp)
                            merged_train_dev_file.write('\n')
                        else:
                            merged_test_file.write(temp)
                            merged_test_file.write('\n')
                        count = count + 1
                        print(temp)
                        print('{}/{}'.format(count, total))
                text_file.close()

random.shuffle(merged_train_dev_file)
random.shuffle(merged_test_file)
merged_test_file.close()
merged_train_dev_file.close()
