import os
import nltk
import re
import numpy as np
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# download NLTK data and initialize tools
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

base_path = "/Users/aimeeco/peer-review-ML-model/data"
sentence_path = os.path.join(base_path, "sentence.pickle")
fileid_path = os.path.join(base_path, "fileid.pickle")
tags_path = os.path.join(base_path, "tags.pickle")

# check paths
print(f"Sentence Path: {sentence_path}")
print(f"FileID Path: {fileid_path}")
print(f"Tags Path: {tags_path}")

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# load data
un_sentences = load_pickle(sentence_path)
fileid = load_pickle(fileid_path)
tags = load_pickle(tags_path)

num_rows = len(un_sentences)
print(f"Number of rows: {num_rows}")

def preprocess():
    splitted = []
    for i in range(num_rows):
        divided = un_sentences[i]
        divided = re.sub(r'\\n', '', divided).strip()
        divided = re.sub(r'[\\]', '', divided)
        divided = re.sub(r'[(),:.";?><]', '', divided)
        divided = re.sub(r'[0-9]*', '', divided).strip()
        divided = re.sub(r" [?\([^)]-+\)]", '', divided)
        splitted.append(divided)
    print(f"Preprocessed {len(splitted)} sentences")
    return splitted

def normal_preprocess():
    norm = []
    for i in range(num_rows):
        divided = un_sentences[i]
        divided = re.sub(r'\\n', '', divided).strip()
        divided = re.sub(r'[\\]', '', divided)
        norm.append(divided)
    print(f"Normal preprocessed {len(norm)} sentences")
    return norm

def listToString(s):
    str1 = " "
    return str1.join(s)

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

# i=0 for normal text and i=1 for processed text
def create_sentence_list(i):
    if i == 1:
        processed_sentences = preprocess()
    else:
        processed_sentences = normal_preprocess()
    sentence_list = []
    for row in range(num_rows):
        sentence_list.append(lemmatize_sentence(porter.stem(processed_sentences[row].lower())))
    print(f"Created sentence list with {len(sentence_list)} sentences")
    return sentence_list

def normal_tag(word):
    sep = []
    ws = word.split(',')
    for w in ws:
        tar = (w.split('-'))[0]
        sep.append(tar.upper())
    return sep

# use create_label(0) to access tag-1, and so on 
def create_label(i):
    label = []
    for row in range(num_rows):
        label.append(normal_tag(tags[row][i]))
    return label

def create_tag_vs_sen(j):
    labels = create_label(j)
    label_dic = {}
    for i in range(num_rows):
        for label in labels[i]:
            if label in label_dic.keys():
                label_dic[label].append(i)
            else:
                label_dic[label] = [i]
    return label_dic

def collect_tags(i):
    return separate_tags(i)

def create_tag_list():
    tag_list = []
    for i in range(num_rows):
        collect_tags(i)
    return tag_list

# returns stopword removed sentences in list [['authors', 'argue'], ['paper', 'due']]
def stop_word_removed_sen():
    li = create_sentence_list(1)
    removed = []
    for l in li:
        wordsList = nltk.word_tokenize(l.lower())
        wordsList_new = [w for w in wordsList if not w in stop_words and len(w) > 2]
        removed.append(wordsList_new)
    return removed

def word_sen():
    li = create_sentence_list(1)
    removed = []
    for l in li:
        wordsList = nltk.word_tokenize(l)
        wordsList_new = [word for word in wordsList if len(word) > 2]
        removed.append(wordsList_new)
    return removed

# i=0 for normal i=1 for tokenized list of words of a sentences
def word_list(i):
    wor_list = []
    if i == 0:
        lis = word_sen()
    else:
        lis = stop_word_removed_sen()
    for l in lis:
        for w in l:
            if w not in wor_list:
                wor_list.append(w)
    return wor_list

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if (token not in stop_words and len(token) > 3)]
    doc = ' '.join(filtered_tokens)
    return doc

# ensure corpus is not empty
corpus = create_sentence_list(1)
print(f"Corpus has {len(corpus)} sentences")
if len(corpus) == 0:
    raise ValueError("Corpus is empty. Please check the input data and preprocessing steps.")

# vectorize the normalization of the corpus
normalize_corpus = np.vectorize(normalize_document, otypes=[str])
norm_corpus = normalize_corpus(corpus)

# make sure all elements in tags are strings before using eval
tags = [str(tag) for tag in tags]

# convert the tags to separate columns
section_labels, aspect_labels, purpose_labels, significance_labels = zip(*[eval(tag) for tag in tags])

df = pd.DataFrame({
    'sentence': un_sentences,
    'normalized_sentence': norm_corpus,
    'file_id': fileid,
    'section_label': section_labels,
    'aspect_label': aspect_labels,
    'purpose_label': purpose_labels,
    'significance_label': significance_labels
})

processed_data_path = '/Users/aimeeco/peer-review-ML-model/data/processed_data.csv'
df.to_csv(processed_data_path, index=False)

print(f"processed data saved to '{processed_data_path}'.")
