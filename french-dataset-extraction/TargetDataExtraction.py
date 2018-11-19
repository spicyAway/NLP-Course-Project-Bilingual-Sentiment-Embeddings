# Reference
# https://machinelearningmastery.com/clean-text-machine-learning-python/
import nltk
# nltk.download('punkt')
# nltk.download('words')
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import words as engwords
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import requests
import string
import re

location = 'Paris'
category = 'accommodation'
language = 'fr'
url = "http://tour-pedia.org/api/getReviews?location="+location+"&category="+category+"&language="+"fr";
res = requests.get(url)

pos_file = open("pos.txt", "a")
neg_file = open("neg.txt", "a")
mix_file = open("mixed.txt", "a")
strong_pos_file = open("strongpos.txt", "a")
strong_neg_file = open("strongneg.txt", "a")

pos_count = 0
neg_count = 0
mix_count = 0
strong_pos_count = 0
strong_neg_count = 0


def basic_nltk_preprocessing(input):
    tokens = word_tokenize(input, language='french')
    # convert to lower case
    tokens = [w.lower() for w in tokens]

    filtered_stripped = tokens

    # for idx, letter in enumerate(tokens):
    #     if letter == '&' and tokens[idx+1] == '#' and tokens[idx+2] == '39' and tokens[idx+3] == ';':
    #         filtered_stripped.append('\'')
    #         print(filtered_stripped)
    #     else:
    #         filtered_stripped.append(letter)

    # remove punctuation from each word
    to_delete = set(string.punctuation) - {'.', ',', '!', ';', '(', ')', '\''}
    stripped = [x for x in filtered_stripped if x not in to_delete]
    if '/' in stripped or '*' in stripped or '@' in stripped or '//' in stripped or '--' in stripped:
        text = [x for x in stripped if x != '/' and x != '*' and x != '@' and x != '//' and x != '--']
    else:
        text = stripped
    # stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    # words = [word for word in tokens if word.isalpha()]
    # filter out stop words
    # stop_words = set(stopwords.words('french'))
    # words = [w for w in words if not w in stop_words]
    # stemming of words
    # text = stemming(words)
    return ' '.join(text)


def stemming(words):
    # stemming of words
    stemmer = SnowballStemmer("french")
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)


#remove reference to urls
#parameter text is a string
def remove_link(text):
    if 'http' in text or 'www' in text:
        text = re.sub(r'(http)\S*|(www)\S*', '', text, flags=re.UNICODE) # no links
        return text
    else:
        return text


def filter_english(text):
    tokens = word_tokenize(text)
    for token in tokens:
        if token in engwords.words():
            return ""
    return text

#replace emphasis on words so helloooooo is the same as hello
#find all characters repeated more than 3 times and replace it with just one
#parameter text is a string
def remove_emphasis(text):
    punc_pattern = re.compile(r'(,\s){2,}')
    text = re.sub(punc_pattern, ', ', text)
    punc_pattern = re.compile(r'(!\s){2,}')
    text = re.sub(punc_pattern, '! ', text)
    punc_pattern = re.compile(r'(\\s){2,}')
    text = re.sub(punc_pattern, '\ ', text)
    punc_pattern = re.compile(r'(-\s){2,}')
    text = re.sub(punc_pattern, '', text)
    return re.sub(r'([a-zA-Z])\1\1\1+', r'\1', text)


# remove emojis, chinese & hebrew characters, whitespaces, and links
#parameter text is a string
def further_text_processing(text):
    text = remove_link(text)
    text = remove_emphasis(text)

    # filters = ['\d+', r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u00BF\u00A1\u00AD\u00AB\u00BB]+',
    #            '\s']
    #
    # tokens = word_tokenize(text)
    # words = []
    # for word in tokens:
    #     for row in filters:
    #         rg = re.compile(row, flags=re.UNICODE)
    #         word = rg.sub(u'', word)
    #     words.append(word)

    return text#' '.join(words)


if res.ok:
    # records = json.dumps(json.loads(res.content))
    res.encoding = 'UTF-8'
    records = res.json()
    for each in records:
        lang = each[u'language']
        polarity = each[u'polarity']
        text = each[u'text'].encode('utf-8').decode('utf-8')
        # word_count = each['wordsCount']
        if lang == 'fr':
            # text = filter_english(text)
            # if text == "":
            #     continue
            # else:
            if '//' in text:
                text = text.replace('//', '')
            if '39' in text:
                text = text.replace('&#39;', "'")

            text1 = basic_nltk_preprocessing(text)
            text = further_text_processing(text1)

            if polarity > 8:
                strong_pos_file.write(text)
                strong_pos_file.write('\n')
                strong_pos_count = strong_pos_count + 1
            elif polarity > 5:
                pos_file.write(text)
                pos_file.write('\n')
                pos_count = pos_count + 1
            elif polarity == 5:
                mix_file.write(text)
                mix_file.write('\n')
                mix_count = mix_count + 1
            elif polarity <= 1:
                strong_neg_file.write(text)
                strong_neg_file.write('\n')
                strong_neg_count = strong_neg_count + 1
            else:
                neg_file.write(text)
                neg_file.write('\n')
                neg_count = neg_count + 1
        else:
            print("program terminated: wrong language data!")
            break


    print('pos:{}'.format(pos_count))
    print('neg:{}'.format(neg_count))
    print('strong_pos:{}'.format(strong_pos_count))
    print('strong_neg:{}'.format(strong_neg_count))
    print('mix:{}'.format(mix_count))

    neg_file.close()
    pos_file.close()
    mix_file.close()
    strong_neg_file.close()
    strong_pos_file.close()

else:
    res.raise_for_status()


