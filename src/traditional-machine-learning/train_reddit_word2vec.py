import os, re
import gzip, json

from gensim.models import Word2Vec

from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize


# Train a word2vec model: https://rare-technologies.com/word2vec-tutorial/
class RedditComments(object):
    def __init__(self, fnames):
        self.covered_fnames = []
        self.fnames = fnames
    
    def __iter__(self):
        for txt_file in self.fnames:
	    if len(self.covered_fnames) % 1000 == 0:
		print "Files covered so far: ", len(self.covered_fnames)
	    self.covered_fnames.append(txt_file)
            with gzip.open(inp_path+'/'+txt_file, 'rb') as f:
                for line in f:
                    dic = json.loads(line)
                    comment = dic["body"]
                    for sentence in comment_to_sents(comment):
                        words = word_tokenize(sentence)
                        # Ignore really short sentences
                        if len(words) > 3:
                            yield words
                            

# Converts line to sentences
def comment_to_sents(comment):
    sentences = sent_tokenize(comment)
    for s in sentences:
        s = re.sub(r'[^a-zA-Z]', " ", s)
        yield s

inp_path = '/home/ndg/arc/reddit/2016/'
my_fnames = []
for fname in os.listdir(inp_path):
    if fname.endswith('.txt.gz'):
        my_fnames.append(fname)
print "Total number of files: ", len(my_fnames)

rc = RedditComments(my_fnames)
model = Word2Vec(rc, size=300, workers=8, min_count=10)
output_path = '/path/beyond-polarization/reddit_trained_allof2016.word2vec'
model.save(output_path)
