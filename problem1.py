# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:52:59 2019

@author: Yanis
"""

import numpy as np
import math
import pandas as pd


#Generic enogh to work for multiclass datasets
def NaiveBayes(dataset, n):
    accuracies = []
    for i in range(n):
        
        X_train, X_test = split_dataframe(dataset, 0.8)

        class_division = {}

        for index, feature_vector in X_train.iterrows():
            feature_vector=list(feature_vector)
            if feature_vector[8] not in class_division:
                class_division[feature_vector[8]] = []   
            class_division[feature_vector[8]].append(feature_vector)
                
        mean_std_tuples = {}
        for label, instances in class_division.items():
            mean_std_tuples[label] = group_mean_std_tuples(instances)
        
        predictions = bulk_predict(mean_std_tuples, X_test)
        #print(predictions)
        accuracy = estimate_accuracy(X_test, predictions)
        accuracies.append(accuracy)
    return accuracies

def group_mean_std_tuples(dataset):
    mean_std_tuples = []
    for vector in zip(*dataset):
        vector = clear_nan_values(vector)
        average = mean(vector)
        std = standard_deviation(average, vector)
        mean_std_tuples.append((average,std))
    del mean_std_tuples[-1] #remove the last label column summaries
    return mean_std_tuples

def mean(numbers):
    return sum(numbers) / float(len(numbers))


def standard_deviation(average, numbers):
    variance = sum([pow(x - average, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# choose y with largest sum(log(p(Xi|y))) + log(p(y)) 
# Normal distribution for number
# Y = { 1/[ σ * sqrt(2π) ] } * e-(x - μ)2/2σ2
# Source: https://stattrek.com/probability-distributions/normal.aspx
def estimate_posterior(x, mean, std):
    exponent = math.exp(-(math.pow(float(x) - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

# for every feature value and the corresponding mean and std use the respective distribution to 
# caculate probabilities of X to belong to any of the chosen classes
def estimate_probabilities_per_class(distribution_parameters, test_vector):
    probabilities = {}
    for label, dist_values in distribution_parameters.items():
        probabilities[label] = 1 #default it to 1
        for i in range(len(dist_values)):
            mean, std = dist_values[i]
            X = test_vector[i]
            if math.isnan(X):
                continue
            probabilities[label] = probabilities[label] * estimate_posterior(X, mean, std)
    return probabilities

#shuffle the dataframe and take the ration*data len for training and the rest for testing
def split_dataframe(dataframe, train_ratio):
    dataframe = dataframe.iloc[np.random.permutation(len(dataframe))]

    trainset_len = int(len(dataframe)*train_ratio)
    trainset = dataframe.iloc[0:trainset_len , : ]
    testset = dataframe.iloc[trainset_len:len(dataframe) , : ]
    
    return trainset,testset
    
    
def clear_nan_values(vector):
    new_vector = []
    for number in vector:
        if not math.isnan(number):
            new_vector.append(number)
    return new_vector

# For given test vector and estimated parameters find the class with the highest probability and assign it to this vector
def predict(distribution_parameters, test_vector):
    probabilities = estimate_probabilities_per_class(distribution_parameters, test_vector)
    predicted_label = None
    max_probability = -1

    for label, probability in probabilities.items():
        if probability > max_probability:
            max_probability = probability
            predicted_label = label
    return predicted_label

#Caclucates the prediction per every row in the test set based on trained data
def bulk_predict(distribution_parameters, test_set):
    predictions = [predict(distribution_parameters, list(feature_test_vector)) for index, feature_test_vector in test_set.iterrows()]
    return predictions

# Estimate % of correct results
def estimate_accuracy(test_set, results):
    correct = 0
    count = 0
    for index, feature_test_vector in test_set.iterrows():
        feature_test_vector = clear_nan_values(feature_test_vector)
        if feature_test_vector[-1] == results[count]:
            correct += 1
        count = count+ 1
    return (correct / float(len(test_set))) * 100.0


# Part 1
pima_indians = pd.read_csv('pima-indians-diabetes.csv')

print(mean(NaiveBayes(pima_indians, 10)))

# Part 2
dataset_manipulated = pima_indians.values

for i in [2, 3, 5, 7]:
    for j in range(len(dataset_manipulated)):
        if dataset_manipulated[j][i] == 0:
            dataset_manipulated[j][i] = np.nan

result=NaiveBayes(pd.DataFrame(dataset_manipulated),10)
print(mean(result))

    # Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)