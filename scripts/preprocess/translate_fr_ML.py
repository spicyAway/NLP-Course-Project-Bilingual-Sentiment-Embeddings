from time import sleep
import copy
from googletrans import Translator
import os
from nltk.tokenize import sent_tokenize

dataset_type = ['train', 'test', 'dev']
sentiment_type = ['neg', 'pos', 'strneg', 'strpos']

for type in dataset_type:
    for sentiment in sentiment_type:

        output_file = os.path.join('datasets', 'trans', 'fr', 'opener_sents', type, 'trans_' + sentiment + '.txt')
        text_file = open(os.path.join('datasets', 'trans', 'fr', 'opener_sents', type, sentiment + '.txt'), "r")
        lines = text_file.read().split('\n')
        print(len(lines))

        translator = Translator()
        count = 0
        timer = 0
        total = len(lines)
        with open(output_file, "a") as text_file:
            for line in lines:
                paragraph = []
                sent_tokenize_list = sent_tokenize(line, language='french')
                for sent in sent_tokenize_list:
                    # sent = 'le bar dédié à la veuve cliquot'
                    translator = Translator()
                    newrow = copy.deepcopy(sent)
                    translation = translator.translate(newrow, dest='en')
                    result = translation.text
                    paragraph.append(result)
                temp = ' '.join([str(x) for x in paragraph])
                text_file.write(temp)
                text_file.write('\n')
                count = count + 1
                timer = timer + 1
                if timer > 30:
                    sleep(120)
                    timer = 0
                print(temp)
                print('{}/{}'.format(count, total))

        text_file.close()