"""
This file contains functions to compute sentiment-related features from SentiwordNet and VADER.
"""

from nltk.corpus import wordnet, sentiwordnet
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def vader_vector(text):
    """
    Extracts VADER scores for the given text.

    Returns
    -------
    list
        positive, negative, and neutral scores computed by VADER
    """
    vader_dict = analyzer.polarity_scores(text)

    return [vader_dict['pos'], vader_dict['neg'], vader_dict['neu']]


def penn_to_wn(tag):
    """
    Convert the PennTreebank tags to simple Wordnet tags.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None


def swn_polarity(tagged_sents):
    """
    Calculates the Sentiwordnet polarity scores for the given comment 'tagged_sents'.

    Parameters
    ----------
    tagged_sents : list of lists where each inner list is a list of tuples (word, tag)

    Returns
    -------
    normalized pos_score, neg_score, obj_score
    """
    word_count = 0
    comment_pos_score = 0.0; comment_neg_score = 0.0; comment_obj_score = 0.0

    for tagged_sentence in tagged_sents:
        for word, tag in tagged_sentence:
            word_count += 1

            wn_tag = penn_to_wn(tag)
            # Skip if pos_tag not in one of these four:
            if wn_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wordnet.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = sentiwordnet.senti_synset(synset.name())

            if swn_synset == None: # Was None for 'blown.s.01' for some reason
                continue

            comment_pos_score += swn_synset.pos_score()
            comment_neg_score += swn_synset.neg_score()
            comment_obj_score += swn_synset.obj_score()

    # Normalize:
    comment_pos_score /= word_count; comment_neg_score /= word_count; comment_obj_score /= word_count

    return comment_pos_score, comment_neg_score, comment_obj_score
