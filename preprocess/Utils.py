# linguistics
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
import string

ntk_stop_words = set(stopwords.words('english'))

def preProcessText_1(text):
    # all to lower case
    text = text.lower()
    # remove square brackets and its content
    text = re.sub('\[.*?\]','', text)
    # remove punctuation 
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    # replace digits in phrases
    text = re.sub('\w*\d\w*', '', text)
    return text

def preProcessText_2(text):
    # remove additional punctuations like italicized/ some latex type quotes extra quotes
    text = re.sub('[‘’“”…]', '', text)
    # remove any \n present in the text
    text = re.sub('\n', '', text)
    # for carriage returns leave some space while replacing
    text = re.sub('\r', ' ', text)
    return text


## ntlk processing
def preProcessText_3(text, stop_words=None):
    if not stop_words:
        stop_words = ntk_stop_words
    # Word tokenizers is used to find the words  
    # and punctuation in a string 
    wordsList = nltk.word_tokenize(text)
    
    # removing stop words from wordList 
    wordsList = [w for w in wordsList if w not in stop_words]
    return wordsList

def posTagger(wordsList):
    #  Using a Tagger. Which is part-of-speech  
    # tagger or POS-tagger.  
    tagged = nltk.pos_tag(wordsList) 
    return tagged

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

lemmatizer = WordNetLemmatizer()

def lemmatizeContent(tagged):
    content = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag)
        content.append(lemma)
    return ' '.join(content)
