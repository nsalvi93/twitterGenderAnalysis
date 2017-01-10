import sys
try:
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
    from nltk.corpus import words
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import roc_curve
    from sklearn.pipeline import Pipeline, FeatureUnion

except ImportError:
    print '[!] You need to install nltk (http://nltk.org/index.html)'

import scipy as sp
import pandas as pd
import numpy as np
import Utilities
import Algorithms
import re
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

# ----------------------------------------------------------------------# ----------------------------------------------------------------------
def _calculate_languages_ratios(text):                                                                                  # checking if tweet is in english

    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios

# ----------------------------------------------------------------------# ----------------------------------------------------------------------

def cleaned_text(text):                                                                                                 # get english words without stopwords

    tokens = wordpunct_tokenize(text)
    text_words = [word.lower() for word in tokens]
    #stopwords_set = stopwords.words('english')

    englishwords_set = words.words()                                                                                    # getting all english words

    textwords_set = set(text_words)
    #proper_nouns =  get_continuous_chunks(text)
    valid_words = textwords_set.intersection(englishwords_set)
    valid_words_without_stopwords = [w for w in valid_words if not w in stopwords.words("english")]                     # removing stop words
    return valid_words_without_stopwords

# ----------------------------------------------------------------------# ----------------------------------------------------------------------

def get_continuous_chunks(text):                                                                                        # for taking names and other proper nouns
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
    return continuous_chunk

# ----------------------------------------------------------------------# ----------------------------------------------------------------------
def detect_language(text):                                                                                              # function to detect language
    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language

# ----------------------------------------------------------------------# ----------------------------------------------------------------------


def cleaningData_addingFeatureVectors(filename):
    valid_wordlist = []  # adding a new column
    data_frame = pd.read_csv(filename, sep=",", header='infer')  # loading csv in pandas dataframe

    data_frame.dropna(how="all", inplace=True)

    data_frame['text'] = data_frame['text'].map(lambda x: re.sub(r'\W+', ' ', x))
    data_frame['description'] = data_frame['description'].str.replace(r'\W+', ' ')

    data_frame = data_frame.loc[data_frame['gender'] != 'unknown']
    data_frame = data_frame[pd.notnull(data_frame['gender'])]

    for index, row in data_frame.iterrows():

        description_and_text = str(row['description']) + " " + str(row['text'])
        language = detect_language(description_and_text)                                                                # put in checking condition for dropping rows
        if (language != "english"):                                                                                     # cleaning data which is not in english
            print("entered")
            data_frame.drop(index, inplace=True)  # dropping rows
            continue

        proper_nouns = get_continuous_chunks(description_and_text)
        valid_words = cleaned_text(description_and_text)
        valid_wordlist.append(', '.join(valid_words))
    data_frame.drop(data_frame.columns[[1, 2, 3, 4, 7, 8, 11, 12, 15, 16, 20, 22, 23, 24, 25]], axis=1,inplace=True)  # removing useless columns
    data_frame['valid_words'] = valid_wordlist
    return data_frame



if __name__ == '__main__':
    #data_frame = cleaningData_addingFeatureVectors("/Users/nishantsalvi/Downloads/twitter.csv")                        # original dataset
    Utilities.pandas_printOptions()
    data_frame = pd.read_pickle("cleaned_removed_unkowns_fullDataset.pkl")                                              # reading from pkl

    Algorithms.testing_randomForests(data_frame)                                                                       # uncomment as required
    Algorithms.testing_NaiveBayes(data_frame)
    Algorithms.testing_logsiticRegression(data_frame)
    Algorithms.testing_supportVectorMachines(data_frame)
