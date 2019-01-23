# -*- coding: utf-8 -*-
"""Simple Demo of a 1-feature Naive Bayes.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import math

data = []
X = []
Y = []
# import the data from the csv.
with open('pima-indians-diabetes.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)
    # remove the header from the data.
    data.pop(0)


'''
def train_and_predict(train_data, test_data):
    # train/build the model based on training set data
    # Calculate p(Positive) & p(negative)
    positive = train_data[train_data[:, 8] == 1]
    negative = train_data[train_data[:, 8] == 0]

    p_positive = 1.0*positive.shape[0]/train_data.shape[0]
    p_negative = 1 - p_positive

    # Calculate mean and variance for all features for both positive and negative samples
    positive_mean = np.mean(positive, axis=0)[:-1]
    positive_var = np.var(positive, axis=0)[:-1]

    negative_mean = np.mean(negative, axis=0)[:-1]
    negative_var = np.var(negative, axis=0)[:-1]

    # to predict on the test data set.
    test_X = np.array(test_data)[:, :-1]
    test_Y = np.array(test_data)[:, -1]

    positive_class = np.sum(np.log(norm.pdf(test_X, positive_mean, np.sqrt(positive_var))), axis=1) + np.log(p_positive)
    negative_class = np.sum(np.log(norm.pdf(test_X, negative_mean, np.sqrt(negative_var))), axis=1) + np.log(p_negative)
    predicted = (positive_class > negative_class)*1
    accuracy = sum(predicted == test_Y)/test_Y.shape[0]
    return accuracy
'''

def train_and_predict(train_data, test_data):
    # train/build the model based on training set data
    # Calculate p(Positive) & p(negative)
    positive = train_data[train_data[:, 8] == 1]
    negative = train_data[train_data[:, 8] == 0]

    p_positive = 1.0*positive.shape[0]/train_data.shape[0]
    p_negative = 1 - p_positive

    # Calculate mean and variance for all features for positive samples
    positive_mean = np.mean(positive[:,:-1], axis=0)
    positive_var = np.var(positive[:,:-1], axis=0)

    # adjust attribute 3 to ignore the zero values
    col3 = positive[:, 2]
    positive_mean[2] = np.mean(col3[col3 != 0])
    positive_var[2] = np.var(col3[col3 != 0])

    # adjust attribute 4 to ignore the zero values
    col4 = positive[:, 3]
    positive_mean[3] = np.mean(col4[col4 != 0])
    positive_var[3] = np.var(col4[col4 != 0])

    # adjust attribute 6 to ignore the zero values
    col6 = positive[:, 5]
    positive_mean[5] = np.mean(col6[col6 != 0])
    positive_var[5] = np.var(col6[col6 != 0])

    # adjust attribute 8 to ignore the zero values
    col8 = positive[:, 7]
    positive_mean[7] = np.mean(col8[col8 != 0])
    positive_var[7] = np.var(col8[col8 != 0])

    # Calculate mean and variance for all features for negative samples
    negative_mean = np.mean(negative[:,:-1], axis=0)
    negative_var = np.var(negative[:,:-1], axis=0)

    # adjust attribute 3 to ignore the zero values
    col3 = negative[:, 2]
    negative_mean[2] = np.mean(col3[col3 != 0])
    negative_var[2] = np.var(col3[col3 != 0])

    # adjust attribute 4 to ignore the zero values
    col4 = negative[:, 3]
    negative_mean[3] = np.mean(col4[col4 != 0])
    negative_var[3] = np.var(col4[col4 != 0])

    # adjust attribute 6 to ignore the zero values
    col6 = negative[:, 5]
    negative_mean[5] = np.mean(col6[col6 != 0])
    negative_var[5] = np.var(col6[col6 != 0])

    # adjust attribute 8 to ignore the zero values
    col8 = negative[:, 7]
    negative_mean[7] = np.mean(col8[col8 != 0])
    negative_var[7] = np.var(col8[col8 != 0])

    # to predict on the test data set.
    test_X = np.array(test_data)[:, :-1]
    test_Y = np.array(test_data)[:, -1]

    p_x_positive = np.log(norm.pdf(test_X, positive_mean, np.sqrt(positive_var)))
    p_x_negative = np.log(norm.pdf(test_X, negative_mean, np.sqrt(negative_var)))
    for index, item in enumerate(test_X):
        if item[2] == 0:
            p_x_positive[index][2] = 0
            p_x_negative[index][2] = 0
        if item[3] == 0:
            p_x_positive[index][3] = 0
            p_x_negative[index][3] = 0
        if item[5] == 0:
            p_x_positive[index][5] = 0
            p_x_negative[index][5] = 0
        if item[7] == 0:
            p_x_positive[index][7] = 0
            p_x_negative[index][7] = 0

    positive_class = np.sum(p_x_positive, axis=1) + np.log(p_positive)
    negative_class = np.sum(p_x_negative, axis=1) + np.log(p_negative)
    predicted = (positive_class > negative_class)*1
    accuracy = sum(predicted == test_Y)/test_Y.shape[0]
    return accuracy

data = np.array(data)
split_idx = int(data.shape[0]*0.2)

total_accuracy = 0.0
total_iteration = 10
for i in range(total_iteration):
    np.random.shuffle(data)
    test_set = data[:split_idx, :]
    train_set = data[split_idx:, :]
    total_accuracy = total_accuracy + train_and_predict(train_set, test_set)
    print("train accuracy "+str(i)+" is ", str(train_and_predict(train_set, train_set)))
    print("test accuracy " + str(i) + " is ", str(train_and_predict(train_set, test_set)))

average_accuracy = total_accuracy/total_iteration
print("accuracy is : ", average_accuracy)
#print("shape of train set is: ", train_set.shape)
#print("shape of test set is: ", test_set.shape)

