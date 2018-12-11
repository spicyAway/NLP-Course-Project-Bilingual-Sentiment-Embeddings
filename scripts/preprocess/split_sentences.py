import nltk

def split(infile, outfile):
    with open(infile, 'rb') as file:
        reviews = file.readlines()

    result = []
    for r in reviews:
        sent_text = nltk.sent_tokenize(r.decode('utf-8'))
        sent_text = [s for s in sent_text if len(s) >= 2]
        result += sent_text

    with open(outfile, 'w') as file:
        for s in result:
            file.write(s + "\n")

dataset_type = ['train', 'dev', 'test']
sentiment_type = ['neg', 'pos', 'strneg', 'strpos']

for type in dataset_type:
    for sentiment in sentiment_type:
        split('datasets/trans/fr/opener_sents/'+type+'/'+sentiment+'.txt', 'datasets/trans/fr/opener_sents/'+type+'/trans_'+sentiment+'.txt')
