import re
from nltk.tokenize import sent_tokenize

def clean_text(text):
    """
    Cleans the given text. Removes URL links. Removes non-ASCII characters.
    """
    text = re.sub(r"http\S+", "", text) # remove URL links
    return ''.join([i if ord(i) < 128 else '' for i in text])


def extract_questions_regex(comment):
    """
    Extracts questions from a given comment using simple regular expression and sentence tokenization.
    Note: gets rid of non-ASCII characters.
    
    Parameters
    ----------
    comment: str
    
    Returns
    -------
    list
        A list of strings where each string corresponds to a question.
    """
    questions = []
    
    clean_comment = re.sub(r'[!.,?]*\?[!.,?]*', '? ', comment) # substitute multiple !??.. with a single "?"
    clean_comment = re.sub(r'\.+\.', '. ', clean_comment) # substitute multiple .... with a single "."
    
    paras = clean_comment.split('\n')

    for para in paras:
        sentences = sent_tokenize(para)
        for sent in sentences:
            sent = sent.strip()
            if sent.endswith('?'): # gets rid of quoting questions: I quit after "do you touch yourself?"
                if sent.startswith('"') or sent.startswith("'"): # avoids sent_tokenize errors
                    continue
                questions.append(sent)
                
    return questions
