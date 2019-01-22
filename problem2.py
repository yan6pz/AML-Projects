# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:52:59 2019

@author: Yanis
"""

# Classification template

# Importing the libraries
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from skimage import transform
import os

np.random.seed(123)

from scipy.io import loadmat
from scipy.stats import norm, bernoulli
from six.moves import urllib


def NaiveBayes(X_train, y_train, X_test, plot = False):
        # prepare model
        class_division = {}
        y_train = list(y_train)

        for index, feature_vector in X_train.iterrows():
            feature_vector=list(feature_vector)
            if y_train[index] not in class_division:
                class_division[y_train[index]] = []   
            class_division[y_train[index]].append(feature_vector)
            
        mean_std_tuples = {}
        for label, instances in class_division.items():
            mean_std_tuples[label] = group_mean_std_tuples(instances)
        
        #plot mean images
        if plot:
            for key, values in mean_std_tuples.items():
                means = [mean for mean, std in values]
                means.append(0)
                plot_mean_image(means, key)
        
        predictions = bulk_predict(mean_std_tuples, X_test, class_division)

        return predictions

def estimate_prior_prob(training_class_set):
    prior_prob = {}
    for i in range(10):
        prior_prob[i] = len(training_class_set[i]) / len(train)
    
    return prior_prob

def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    width = int(math.sqrt(len(image)))
    two_d = (np.reshape(image, (width, width))).astype(np.uint8)
    plt.title("Label: %s" % label)
    plt.imshow(two_d,cmap=plt.cm.gray, origin='upper', interpolation='nearest')
    return plt
    
def plot_mean_image(vector,label):
    view_image(vector,label).show()
    
def group_mean_std_tuples(dataset):
    mean_std_tuples = []
    for vector in zip(*dataset):
        average = mean(vector)
        std = standard_deviation(average, vector)
        mean_std_tuples.append((average,std))
    del mean_std_tuples[-1] #remove the last label column summaries
    return mean_std_tuples

def mean(numbers):
    return sum(numbers) / float(len(numbers))


def standard_deviation(average, numbers):
    variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance) +1

# choose y with largest sum(log(p(Xi|y))) + log(p(y)) 
# Normal distribution for number
# Y = { 1/[ σ * sqrt(2π) ] } * e-(x - μ)2/2σ2
# Source: https://stattrek.com/probability-distributions/normal.aspx
def estimate_posterior(x, mean, std, prior):
    #print("X: {} , mean: {}, std: {}".format(x,mean,std))
    #log_probability = np.log(norm.pdf(x,mean,std))+ np.log(prior)
    exponent = math.exp(-(math.pow(float(x) - mean, 2) / (2 * math.pow(std, 2)))) +1
    value =1 / (math.sqrt(2 * math.pi) * std) +1
    return math.log(value * exponent) + math.log(prior + 1)
    #return log_probability

# for every feature value and the corresponding mean and std use the respective distribution to 
# caculate probabilities of X to belong to any of the chosen classes
def estimate_probabilities_per_class(distribution_parameters, test_vector, prior_probabilities):
    probabilities = {}
    for label, dist_values in distribution_parameters.items():
        probabilities[label] = 0 #default it to 1
        for i in range(len(dist_values)):
            mean, std = dist_values[i]
            X = test_vector[i]
            probabilities[label] = probabilities[label] + estimate_posterior(X, mean, std, prior_probabilities[label])
    return probabilities

#shuffle the dataframe and take the ration*data len for training and the rest for testing
def split_dataframe(dataframe, train_ratio):
    dataframe = dataframe.iloc[np.random.permutation(len(dataframe))]

    trainset_len = int(len(dataframe)*train_ratio)
    trainset = dataframe.iloc[0:trainset_len , : ]
    testset = dataframe.iloc[trainset_len:len(dataframe) , : ]
    
    return trainset,testset
    

# For given test vector and estimated parameters find the class with the highest probability and assign it to this vector
def predict(distribution_parameters, test_vector, prior_probabilities):
    probabilities = estimate_probabilities_per_class(distribution_parameters, test_vector, prior_probabilities)
    predicted_label = None
    max_probability = -1

    for label, probability in probabilities.items():
        if probability > max_probability:
            max_probability = probability
            predicted_label = label
    return predicted_label

#Caclucates the prediction per every row in the test set based on trained data
def bulk_predict(distribution_parameters, test_set, class_division):
    prior_probabilities = estimate_prior_prob(class_division)
    #print(prior_probabilities)
    predictions = [predict(distribution_parameters, list(feature_test_vector), prior_probabilities) for index, feature_test_vector in test_set.iterrows()]
    return predictions

# Estimate % of correct results
def estimate_accuracy(test_set, results):
    correct = 0
    count = 0
    for feature_test_vector in test_set:
        if feature_test_vector == results[count]:
            correct += 1
        count = count+ 1
    return (correct / float(len(test_set)))

def bernoulli_fitting(X,y):
    alpha = 1.0
    count_sample = X.shape[0]
    separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
    class_log_prior = [np.log(len(i) / count_sample) for i in separated]
    count = np.array([np.array(i).sum(axis=0) for i in separated]) + alpha
    smoothing = 2 * alpha
    n_doc = np.array([len(i) + smoothing for i in separated])
    feature_prob = count / n_doc[np.newaxis].T

    return class_log_prior,feature_prob

#per each items in the test set estimate the posterior
def predict_log_proba(X, class_log_prior,feature_prob):
    class_probabilities = []
    for x in X:
        #bernoulli.pmf(x, np.log(feature_prob))
        posterior_prob = np.log(feature_prob) * x 
        sum_posterior = posterior_prob.sum(axis=1)
        class_probabilities.append(sum_posterior + class_log_prior)
        
    return class_probabilities
#estimate the class probabilities and get the class with the largest one 
def bernoulli_predicting(X_test,class_log_prior,feature_prob ):
    class_probabilities = predict_log_proba(X_test,class_log_prior,feature_prob)
    return np.argmax(class_probabilities, axis=1)

mnist_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"

exists = os.path.isfile(mnist_path)
if not exists:
    response = urllib.request.urlopen(mnist_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
#number_of_items = 56000
#features = pd.DataFrame(np.array(mnist["data"][:number_of_items,:], 'int16'))
#labels = pd.DataFrame(np.array(mnist["target"][:number_of_items], 'int'))

features = pd.DataFrame(np.array(mnist["data"], 'int16'))
labels = pd.DataFrame(np.array(mnist["target"], 'int'))

data = pd.concat([labels, features], axis=1)
train, test = train_test_split(data, test_size=0.2)

train_x = np.array(train.iloc[:, 1:])
test_x = np.array(test.iloc[:, 1:])
train_y = np.array(train.iloc[:, 0])
test_y = np.array(test.iloc[:, 0])


target_train_y = train_y.astype(np.uint8)
target_test_y = test_y.astype(np.uint8)


def bounding_rescale_strech_image(image):
    initial_size = (28, 28)
    target_size = (20, 20)
    img_matrix = np.array(image).reshape((-1, 1, 28, 28)).astype(np.uint8)
    img = Image.fromarray(img_matrix[0][0])
    threshold_img = img.point(lambda x: 0 if x < 128 else 255, '1')
    img_reshaped = np.reshape(np.array(threshold_img), initial_size)
    row= np.unique(np.nonzero(img_reshaped)[0])
    col = np.unique(np.nonzero(img_reshaped)[1])
    scale_image = img_reshaped[min(row):max(row), min(col):max(col)]
    image_data_new = transform.resize(scale_image, target_size)
    return (np.array(image_data_new).astype(np.uint8))


train_modified = np.apply_along_axis(bounding_rescale_strech_image, axis=1, arr=train_x)
test_modified = np.apply_along_axis(bounding_rescale_strech_image, axis=1, arr=test_x)

train_final = np.reshape(train_modified, (train_modified.shape[0], 400))
test_final = np.reshape(test_modified, (test_modified.shape[0], 400))


def do_naive_bayes_normal(training_x,training_y, test_x, test_y,case, plot= False):
    result = NaiveBayes(training_x,training_y, test_x, plot)

    acc = estimate_accuracy(test_y, result)
    print("Accuracy achieved for " + case + " is " + str(acc * 100) + "%")
    
def do_naive_bayes_bernoulli(training_x, training_y, test_x, test_y, case):
    log_prior, feature_probability = bernoulli_fitting(training_x, training_y)
    result = bernoulli_predicting(test_x, log_prior, feature_probability)
    acc = estimate_accuracy(test_y, result)
    print("Accuracy achieved for " + case + " is " + str(acc * 100) + "%")

#training_set_final = pd.concat([pd.DataFrame(train_final), pd.DataFrame(target_train_y)], axis=1)

do_naive_bayes_normal(pd.DataFrame(train_x), train_y, pd.DataFrame(test_x),test_y,  "untouched_gausian", True)
do_naive_bayes_normal(pd.DataFrame(train_final), target_train_y, pd.DataFrame(test_final), target_test_y, "bounded_stretched_gausian")
do_naive_bayes_bernoulli(train_x, train_y, test_x, test_y, "untouched_bernoulli")
do_naive_bayes_bernoulli(train_final, target_train_y, test_final, target_test_y, "bounded_stretched_bernoulli")

def do_random_forest(d, n, train_x1, train_y1, test_x1, test_y1, case):
    random_forest_model = RandomForestClassifier(max_depth=d, random_state=0, n_estimators=n)
    random_forest_model.fit(train_x1, train_y1)
    preiction = random_forest_model.predict(test_x1)
    acc = estimate_accuracy(test_y1, preiction)
    print("Accuracy achieved for " + case + " is " + str(acc * 100) + "%")


#do_random_forest(4, 10, train_x, train_y, test_x, test_y, "4/10 untouched")
#do_random_forest(4, 30, train_x, train_y, test_x, test_y, "4/30 untouched")
#do_random_forest(4, 10, train_final, target_train_y, test_final, target_test_y, "4/10 bounded and stretched")
#do_random_forest(4, 30, train_final, target_train_y, test_final, target_test_y, "4/30 bounded and stretched")

#do_random_forest(16, 10, train_x, train_y, test_x, test_y, "16/10 untouched")
#do_random_forest(16, 30, train_x, train_y, test_x, test_y, "16/30 untouched")

#do_random_forest(16, 10, train_final, target_train_y, test_final, target_test_y, "16/10 bounded and stretched")
#do_random_forest(16, 30, train_final, target_train_y, test_final, target_test_y, "16/30 bounded and stretched")