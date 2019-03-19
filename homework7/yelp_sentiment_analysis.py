import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from string import punctuation
from operator import itemgetter

from sklearn.feature_extraction.text import CountVectorizer


def plot_word_distribution(occurrences):
    rank = range(len(occurrences))
    plt.figure()
    plt.scatter(rank, occurrences)
    plt.xlabel('Word Rank')
    plt.ylabel('Word Count')
    plt.grid()
    plt.show()


def word_count(data, number_of_words, stop_words, to_print=False):
    words = {}

    words_gen = (word.strip(punctuation).lower() for review in data
                                             for word in review.split() if word not in stop_words)

    for word in words_gen:
        words[word] = words.get(word, 0) + 1


    top_words = sorted(words.items(), key=itemgetter(1), reverse=True)

    if number_of_words:
        top_words = top_words[:number_of_words]

    if to_print:
        i = 0
        for word, frequency in top_words:
            print("%s %d %d" % (word, frequency, i))
            i += 1


    return dict(top_words)


def max_min_document_frequency(data):
    '''
    min_df - ignore terms having document frequency lower than 1%
    max_df - ignore terms having document frequency higher than 80%
    '''
    vectorizer = CountVectorizer(min_df=0.01, max_df=0.8, stop_words='english')
    training_set = vectorizer.fit_transform(data)
    return vectorizer.get_stop_words(), vectorizer, training_set



def plot_histograms(data):
    plt.hist(data['text length'], bins=50)
    plt.ylabel('Number of words')
    plt.show()

def truncate(content, length=200, suffix='...'):
    content_array = content.split(' ')
    if len(content_array) <= length:
        return content
    else:
        return ' '.join(content_array[:length]) + suffix

def print_nearest_neibors(content, distance):
    text = truncate(content)
    print(len(text.split(' ')))
    print("%f %s" % (distance, text))

yelp_dataset = pd.read_csv('yelp_2k.csv')


X = yelp_dataset['text']
y = yelp_dataset['stars']

##################################################################################

#Part 1: Preprocessing with Bag-of-Words

##################################################################################

#words_dictionary = word_count(X, None, {})
#plot_word_distribution(words_dictionary.values())


#words_dictionary_filtered = word_count(X, 500, {})
#plot_word_distribution(words_dictionary_filtered.values())


stop_words, vectorizer, training_set = max_min_document_frequency(X)
words_dictionary_preprocessed = word_count(X, None, stop_words, False)
plot_word_distribution(words_dictionary_preprocessed.values())

#print(X.head())
bag_of_words = []
counter = 0
for review in X:
    bag_of_words.append(list(word.strip(punctuation).lower()
     for word in review.split() if word not in stop_words))

#print(bag_of_words[0])
#print(bag_of_words[1])


##################################################################################

#Part 2: Text-Retrieval

##################################################################################

n = 5
docs_new = ['Horrible customer service']
X_new_doc = vectorizer.transform(docs_new)

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=n, metric='cosine')
neigh.fit(training_set)
distances, indexes = neigh.kneighbors(X_new_doc, return_distance=True)
count = 0

for i in indexes[0]:
    print_nearest_neibors(X.iloc[i], distances[0][count])
    count +=1

'''
yelp_dataset['text length'] = yelp_dataset['text'].apply(len)
print(yelp_dataset.head())




one_star_reviews = yelp_dataset[(yelp_dataset['stars'] == 1)]
five_star_reviews = yelp_dataset[(yelp_dataset['stars'] == 5)]
plot_histograms(one_star_reviews)
plot_histograms(five_star_reviews)
'''
