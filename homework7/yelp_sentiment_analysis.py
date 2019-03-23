import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from operator import itemgetter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def plot_word_distribution(occurrences, title):
    rank = range(len(occurrences))
    plt.figure()
    plt.scatter(rank, occurrences)
    plt.xlabel('Word Rank')
    plt.ylabel('Word Count')
    plt.grid()
    plt.savefig("distribution_" + title)
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
    vectorizer = CountVectorizer(min_df=3, max_df=0.42, stop_words='english')
    training_set = vectorizer.fit_transform(data)
    return vectorizer.get_stop_words(), vectorizer, training_set



def plot_histograms(predictions, threshold):
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=50)
    plt.hist(predictions[predictions[:,0] >= threshold], **kwargs)
    plt.hist(predictions[predictions[:,0] < threshold], **kwargs)
    plt.ylabel('Count of predictions in bucket')
    plt.xlabel('Predicted Score')
    plt.title("Histogram of Predicted Scores")
    plt.savefig('hist_predicted_scores')
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


def get_n_neighbors(data, search_query, n):
    X_new_doc = vectorizer.transform(search_query)
    neigh = NearestNeighbors(n_neighbors=n, metric='cosine')
    neigh.fit(data)
    distances, indexes = neigh.kneighbors(X_new_doc, return_distance=True)

    return distances, indexes

def plot_roc_curve(fpr, tpr, log_roc_auccuracy):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % log_roc_auccuracy)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curve')
    plt.show()


yelp_dataset = pd.read_csv('yelp_2k.csv')


X = yelp_dataset['text']
y = yelp_dataset['stars']

##################################################################################

#Part 1: Preprocessing with Bag-of-Words

##################################################################################

#Initial word distribbution plot
words_dictionary = word_count(X, None, {})
plot_word_distribution(words_dictionary.values(), "initial")

#plot first 500 word distribution and oserve the curve
#words_dictionary_filtered = word_count(X, 500, {})
#plot_word_distribution(words_dictionary_filtered.values())

#apply min and max document frequency and remove stop words
stop_words, vectorizer, training_set = max_min_document_frequency(X)
words_dictionary_preprocessed = word_count(X, None, stop_words, False)
plot_word_distribution(words_dictionary_preprocessed.values(), "stop_words_removed")

#construct bag of words representation
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

#find first 5 neighors
distances, indexes = get_n_neighbors(training_set, docs_new, n)
count = 0
for i in indexes[0]:
    print_nearest_neibors(X.iloc[i], distances[0][count])
    count += 1

#find all of the distances. After the 700th we see all the distances are 1(no similarity at all)
distances, indexes = get_n_neighbors(training_set, docs_new, 2000)
plot_word_distribution(distances[0], "cosine_distances")


##################################################################################

#Part 3: Classification with Logistic Regression

##################################################################################


#split into train and test sets. Fit Log Reg model
X_train, X_test, y_train, y_test = train_test_split(training_set, y, test_size=0.1)
log_regression = LogisticRegression(random_state=0, solver='liblinear',
                         multi_class='ovr').fit(X_train, y_train)  #liblinear for small binary problems

#predit and get accuracy
predictions = log_regression.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions)) #reaching 92% accuracy

#Plot a histogram of the scores on the training data xlabel -predicted prob ylabel- count of predictions
plot_histograms(log_regression.predict_proba(X_train), 0.4)


#ROC Curve
log_roc_accuracy = roc_auc_score(y_test, log_regression.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test.values, log_regression.predict_proba(X_test)[:,1], pos_label=5)

plot_roc_curve(fpr,tpr,log_roc_accuracy)
