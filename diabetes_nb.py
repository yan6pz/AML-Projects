# -*- coding: utf-8 -*-
"""Apply Naive Bayes on Diabetes Classification

"""

import numpy as np
from scipy.stats import norm
import csv


# The following function will train & predict using Naive Bayes of Normal Distribution Model.
# This function will ignore the zero values on column 3,4,6,8 as missing data.
def train_and_predict_ignore_o(train_data, test_data):
    # train/build the model based on training set data.
    # Noted: the last column of both train_data and test_data are the label!

    # Calculate p(Positive) & p(negative)
    positive = train_data[train_data[:, 8] == 1]
    negative = train_data[train_data[:, 8] == 0]

    p_positive = 1.0*positive.shape[0]/train_data.shape[0]
    p_negative = 1 - p_positive

    # adjust attribute 3,4,6,8 to ignore the zero values
    positive[(positive[:, 2] == 0), 2] = np.nan  # filter out attribute 3
    positive[(positive[:, 3] == 0), 3] = np.nan  # filter out attribute 4
    positive[(positive[:, 5] == 0), 5] = np.nan  # filter out attribute 6
    positive[(positive[:, 7] == 0), 7] = np.nan  # filter out attribute 8

    # Calculate mean and variance for all features for positive samples
    positive_mean = np.nanmean(positive[:,:-1], axis=0)
    positive_var = np.nanvar(positive[:,:-1], axis=0)

    # adjust attribute 3,4,6,8 to ignore the zero values
    negative[(negative[:, 2] == 0), 2] = np.nan  # filter out attribute 3
    negative[(negative[:, 3] == 0), 3] = np.nan  # filter out attribute 4
    negative[(negative[:, 5] == 0), 5] = np.nan  # filter out attribute 6
    negative[(negative[:, 7] == 0), 7] = np.nan  # filter out attribute 8

    # Calculate mean and variance for all features for negative samples
    negative_mean = np.nanmean(negative[:,:-1], axis=0)
    negative_var = np.nanvar(negative[:,:-1], axis=0)

    # to predict on the test data set.
    test_X = np.array(test_data)[:, :-1]
    test_Y = np.array(test_data)[:, -1]

    p_x_positive = np.log(norm.pdf(test_X, positive_mean, np.sqrt(positive_var)))
    p_x_negative = np.log(norm.pdf(test_X, negative_mean, np.sqrt(negative_var)))

    # ignore the features with zero values for both classes
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
    return accuracy*100


# The following function will train & predict using Naive Bayes of Normal Distribution Model.
# This function will NOT ignore the zero values on column 3,4,6,8 as missing data.
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
    return accuracy*100


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

data = np.array(data)
split_idx = int(data.shape[0]*0.2)

total_accuracy = 0.0
total_iteration = 10
for i in range(total_iteration):
    np.random.shuffle(data)
    test_set = data[:split_idx, :]
    train_set = data[split_idx:, :]
    total_accuracy = total_accuracy + train_and_predict_ignore_o(train_set, test_set)

average_accuracy = total_accuracy/total_iteration
print("Averaged accuracy of cross validation (ignoring zero values) is : ", average_accuracy, "%")

total_accuracy = 0.0
average_accuracy = 0.0
for i in range(total_iteration):
    np.random.shuffle(data)
    test_set = data[:split_idx, :]
    train_set = data[split_idx:, :]
    total_accuracy = total_accuracy + train_and_predict(train_set, test_set)

average_accuracy = total_accuracy/total_iteration
print("Averaged accuracy of cross validation is : ", average_accuracy, "%")

