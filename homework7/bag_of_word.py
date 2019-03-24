import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def takeSecond(elem):
    return elem[1]

def vectorize_bagOfWd(X, maxDf=0.42, minDf=2):
    plt.figure()
    vectorizer = CountVectorizer()
    bagOfWord = vectorizer.fit_transform(X)
    voc_list = vectorizer.vocabulary_.keys()
    wordRank = [[index, item] for index, item in enumerate(bagOfWord.sum(axis=0).tolist()[0])]
    wordRank.sort(key=takeSecond, reverse=True)
    plt.scatter(range(len(wordRank)), np.array(wordRank)[:, 1])

    #for i in range(10):
    #    print(list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(wordRank[i][0])],' : ', wordRank[i][1])

    # stop word based on frequency, and remove those words that only occurred once.
    vectorizer_cutoff = CountVectorizer(max_df=maxDf, min_df=minDf)
    bagOfWord_swRemoved = vectorizer_cutoff.fit_transform(X)
    voc_list_swRemoved = vectorizer_cutoff.vocabulary_.keys()
    stop_word = set(voc_list)-set(voc_list_swRemoved)
    print("Stop word list size: ", len(stop_word), "\n Stop word list:\n", ','.join(stop_word))
    wordRank_swRemoved = [[index, item] for index, item in enumerate(bagOfWord_swRemoved.sum(axis=0).tolist()[0])]
    wordRank_swRemoved.sort(key=takeSecond, reverse=True)
    plt.figure()
    plt.scatter(range(len(wordRank_swRemoved)), np.array(wordRank_swRemoved)[:, 1])
    return bagOfWord_swRemoved, vectorizer_cutoff, stop_word


def text_retrieval(vectorizer, query):
    neigh = NearestNeighbors(metric="cosine")
    neigh.fit(bagOfWord_vectors.toarray(),Y)
    print("Your query is: ", query)
    query_vector = vectorizer.transform([query])
    query_distance, query_results = neigh.kneighbors(query_vector.toarray(), 5)
    print("5 closest results: ")
    for idx, item in enumerate(query_results[0]):
        print("review #",idx+1, ", distance: ", '%.4f'%query_distance[0,idx], " : ", X[query_results[0,idx]][0:199])

def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(solver='liblinear', max_iter=100)
    clf.fit(X_train, y_train)
    predicted_accurary = clf.score(X_test, y_test)
    train_accuracy = clf.score(X_train, y_train)
    print("Test accuracy is: ", predicted_accurary, " Train accuracy is: ", train_accuracy)
    return clf

def plot_histogram(logistic_clf, X_train, y_train):
    predicted_proba = logistic_clf.predict_proba(X_train)
    plt.hist([predicted_proba[y_train=='1'][:,1],predicted_proba[y_train=='5'][:,1]], bins=100, histtype='stepfilled')

    false_positive = predicted_proba[y_train=='1'][predicted_proba[y_train=='1'][:,1]>=0.5]
    false_negative = predicted_proba[y_train=='5'][predicted_proba[y_train=='5'][:,1]<=0.5]
    print("case/s incorrectly predicted to positive review('5') but actually negative('1'): ", ", ".join(false_positive[:,1].astype(str)))
    print("case/s incorrectly predicted to negative review('1') but actually positive('5'): ", ", ".join(false_negative[:,1].astype(str)))

    max(predicted_proba[y_train=='1'][:,1])
    print("Top 10 largest predicted probability for negative reviews:\n",
          "\n ".join(np.sort(predicted_proba[y_train=='1'][:,1])[-10:].astype(str)))

    min(predicted_proba[y_train=='5'][:,1])
    print("Top 10 smallest predicted probability for positive reviews:\n",
          "\n ".join(np.sort(predicted_proba[y_train=='5'][:,1])[:10].astype(str)))


def predict_new_threshold(model, data_X, threshold):
    predicted_p = model.predict_proba(data_X)
    predicted = np.ones(len(data_X)).astype(int).astype(str)
    predicted[predicted_p[:,1]>=threshold]='5'
    return predicted


def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


data = []
with open("./homework7/yelp_2k.csv") as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        data.append(row)

data = np.array(data[1:])
X = [x.lower() for x in data[:, 5]]
Y = data[:, 3]

bagOfWord_vectors, vectorizer, stopWords = vectorize_bagOfWd(X, 0.42, 3)
#the following line get the most frequent #11732 word (which is 'the')
#list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(11732)]
text_retrieval(vectorizer, "Horrible customer service")


X_train, X_test, y_train, y_test = train_test_split(bagOfWord_vectors.toarray(), Y, test_size=1/10)
#debug
#combined = np.zeros((bagOfWord_swRemoved.shape[0], bagOfWord_swRemoved.shape[1]+1))
#combined[:,:-1] = bagOfWord_swRemoved.toarray()
#combined[:,-1] = list(range(len(X)))
#X_train, X_test, y_train, y_test = train_test_split(combined, Y, test_size=1/10)


logistic_clf = logistic_regression(X_train, X_test, y_train, y_test)
plot_histogram(logistic_clf, X_train, y_train)

train_predicted_thres_updated = predict_new_threshold(logistic_clf, X_train, 0.63)
train_accuracy = sum(train_predicted_thres_updated == y_train) / len(y_train)
print("Training accuracy with threshold changed: ", train_accuracy*100, "%")

test_predicted_thres_updated = predict_new_threshold(logistic_clf, X_test, 0.63)
test_accuracy = sum(test_predicted_thres_updated == y_test) / len(y_test)
print("Test accuracy with threshold changed: ", test_accuracy*100, "%")


#fpr, tpr, thresholds = metrics.roc_curve(y_train, predicted_proba[:,1], pos_label='5',drop_intermediate=False)
#fpr, tpr, thresholds = metrics.roc_curve(y_train, logistic_clf.decision_function(X_train), pos_label='5', drop_intermediate=False)
#roc_auc = metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(y_test, logistic_clf.predict_proba(X_test)[:,1], pos_label='5',drop_intermediate=False)
#fpr, tpr, thresholds = metrics.roc_curve(y_test, logistic_clf.decision_function(X_test), pos_label='5')
roc_auc = metrics.auc(fpr, tpr)
plot_roc(fpr, tpr, roc_auc)

