import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


yelp_dataset = pd.read_csv('yelp_2k.csv')


X = yelp_dataset['text']
y = yelp_dataset['stars']

def plot_histograms(data):
    plt.hist(data['text length'], bins=50)
    plt.ylabel('Number of words')
    plt.show()

yelp_dataset['text length'] = yelp_dataset['text'].apply(len)
print(yelp_dataset.head())

one_star_reviews = yelp_dataset[(yelp_dataset['stars'] == 1)]
five_star_reviews = yelp_dataset[(yelp_dataset['stars'] == 5)]
plot_histograms(one_star_reviews)
plot_histograms(five_star_reviews)
