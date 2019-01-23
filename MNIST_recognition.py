# -*- coding: utf-8 -*-
"""Simple Demo of a 1-feature Naive Bayes.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import bernoulli
from sklearn.ensemble import RandomForestClassifier
from mnist import MNIST
import cv2


def stretch_bounding_box(single_image_data):
    single_image_data = single_image_data.reshape((28, 28))
    vertical_min = np.nonzero(single_image_data)[0].min()
    vertical_max = np.nonzero(single_image_data)[0].max()
    horizon_min = np.nonzero(single_image_data)[1].min()
    horizon_max = np.nonzero(single_image_data)[1].max()
    return cv2.resize(single_image_data[vertical_min: vertical_max+1, horizon_min:horizon_max+1], (20, 20)).reshape(400)


class NaiveBayesNormalDistr:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon
        self.p_digit_class = []
        self.predicted = []
        self.digits = []
        self.digits_mean = []
        self.digits_var = []
        self.p = []

    def train(self, train_data, train_labels):
        self.digits = []
        self.p = []
        self.digits_mean = []
        self.digits_var = []
        # separate training data into different classes (digits)
        for i in range(10):
            self.digits.append(train_data[train_labels[:] == i])
            # Calculate p(digits 0,1,2,....9)
            self.p.append(1.0 * self.digits[i].shape[0] / train_data.shape[0])  # len(p) is 10
            # Calculate mean and variance for all features for all classes(digits)
            self.digits_mean.append(np.mean(self.digits[i], axis=0))  # each digits_mean[i] shape is (784,1)
            self.digits_var.append(np.var(self.digits[i], axis=0))  # each digits_var[i] shape is (784,1)
        self.digits_var = np.array(self.digits_var)
        self.digits_var += self.epsilon * self.digits_var.max()

    def predict(self, test_data):
        self.p_digit_class = []
        self.predicted = []
        if self.digits_mean == [] or self.digits_var ==[] or self.p == []:
            print("Fit your model to training data first")
            return []

        for i in range(10):
            normpdf = norm.pdf(test_data, self.digits_mean[i], np.sqrt(self.digits_var[i]))
            p_post = np.sum(np.log(normpdf), axis=1) + np.log(self.p[i])
            self.p_digit_class.append(p_post)
        self.p_digit_class = np.array(self.p_digit_class)
        self.predicted = np.argmax(self.p_digit_class, axis=0)
        return self.predicted

    def get_accuracy(self, test_label):
        if len(self.predicted) == 0:
            print("Run predict() on your test data first")
            return 0
        elif len(self.predicted)!=len(test_label):
            print("Your test label shape mismatch the shape of your prediction data")
            return 0
        accuracy = sum(self.predicted == test_label) / len(test_label)
        return accuracy

    def plot_all_digits_mean(self):
        if len(self.digits_mean) > 0:
            # convert each digit mean from [10,784] to [10,28,28] (or [10,400] to [10,20,20])
            digit_mean_to_plot = np.array(self.digits_mean)
            image_size = int(np.sqrt(digit_mean_to_plot.shape[1]))
            digit_mean_to_plot = digit_mean_to_plot.reshape((digit_mean_to_plot.shape[0],image_size,image_size))
            digit_mean_to_plot = (digit_mean_to_plot*255).astype(int)

            mainFigure = plt.figure(figsize=(10, 8))
            columns = 5
            rows = 2
            for i in range(1, columns * rows + 1):
                mainFigure.add_subplot(rows, columns, i)
                plt.imshow(digit_mean_to_plot[i-1], cmap='gray')
            plt.show()
        else:
            print("Train your model with data first.")

class NaiveBayesBernoulli:
    def __init__(self):
        self.predicted = []
        self.digits = []
        self.p = []
        self.p_digit_class = []
        self.digits_p_ink = []

    def train(self, train_data, train_labels):
        self.digits = []
        self.p = []
        self.digits_p_ink = []
        # separate training data into different classes (digits)
        for i in range(10):
            self.digits.append(train_data[train_labels[:] == i])
            # Calculate p(digits 0,1,2,....9)
            self.p.append(1.0 * self.digits[i].shape[0] / train_data.shape[0])  # len(self.p) is 10
            # Count each ink pixels to calculate p(ink|C), using plus-one smoothing
            # Count of ink pixel +1 / Total images of such digit + number of ink pixels (dimension of row sample)
            p_ink = (np.sum(self.digits[i], axis=0) + 1) / (self.digits[i].shape[0] + train_data.shape[1])
            self.digits_p_ink.append(p_ink)
        self.digits_p_ink = np.array(self.digits_p_ink)

    def predict(self, test_data):
        self.p_digit_class = []
        self.predicted = []
        if self.p == [] or self.digits_p_ink == []:
            print("Fit your model to training data first")
            return []

        for i in range(10):
            berpmf = bernoulli.pmf(test_data, self.digits_p_ink[i])
            p_post = np.sum(np.log(berpmf), axis=1) + np.log(self.p[i])
            self.p_digit_class.append(p_post)

        self.p_digit_class = np.array(self.p_digit_class)
        self.predicted = np.argmax(self.p_digit_class, axis=0)
        return self.predicted

    def get_accuracy(self, test_label):
        if len(self.predicted) == 0:
            print("Run predict() on your test data first")
            return 0
        elif len(self.predicted) != len(test_label):
            print("Your test label shape mismatch the shape of your prediction data")
            return 0
        accuracy = sum(self.predicted == test_label) / len(test_label)
        return accuracy


def train_and_validate_randomforest(train_data, train_labels, test_data, test_label, n_trees, depth):
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
    clf.fit(train_data, train_labels)
    predicted = clf.predict(test_data)
    accuracy = sum(predicted == test_label)/test_label.shape[0]
    return accuracy

mndata = MNIST('./MNIST')
mndata.gz = True
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# filter out the mid grey pixels and convert it into binary picture
ink_threshold = 255*0.5
images = np.array(images, dtype='uint8')
images[images[:] < ink_threshold] = 0
images[images[:] >= ink_threshold] = 1  # mark it as ink pixel
labels = np.array(labels, dtype='uint8')

test_images = np.array(test_images, dtype='uint8')
test_images[test_images[:] < ink_threshold] = 0
test_images[test_images[:] >= ink_threshold] = 1  # mark it as ink pixel
test_labels = np.array(test_labels, dtype='uint8')


# produce the stretched images for train and test set
stretched_image_map = map(stretch_bounding_box, images)
stretched_image = np.array(list(stretched_image_map))

stretched_test_image_map = map(stretch_bounding_box, test_images)
stretched_test_image = np.array(list(stretched_test_image_map))

# use Naive Bayes Normal D to train and predict on untouched images:
nb_normal = NaiveBayesNormalDistr(1e-1)
nb_normal.train(images, labels)
_ = nb_normal.predict(test_images)
print("Naive Bayes - normal distribution accuracy on untouched data: ", nb_normal.get_accuracy(test_labels))
# use Naive Bayes Normal D to train and predict on stretched images:
nb_normal_stretched = NaiveBayesNormalDistr(1e-1)
nb_normal_stretched.train(stretched_image, labels)
_ = nb_normal_stretched.predict(stretched_test_image)
print("Naive Bayes - normal distribution accuracy on untouched data: ", nb_normal_stretched.get_accuracy(test_labels))

# use Naive Bayes Bernoulli to train and predict on untouched images:
nb_bernoulli = NaiveBayesBernoulli()
nb_bernoulli.train(images, labels)
_ = nb_bernoulli.predict(test_images)
print("Naive Bayes - bernoulli accuracy on untouched data: ", nb_bernoulli.get_accuracy(test_labels))
# use Naive Bayes Bernoulli to train and predict on stretched images:
nb_bernoulli_stretched = NaiveBayesBernoulli()
nb_bernoulli_stretched.train(stretched_image, labels)
_ = nb_bernoulli_stretched.predict(stretched_test_image)
print("Naive Bayes - bernoulli accuracy on stretched data: ", nb_bernoulli_stretched.get_accuracy(test_labels))

# RANDOM FOREST - UNTOUCHED DATA
# use Random forest with setting of trees = 10 and depth = 4
print("Untouched data - Random Forest (10 Trees, 4 Depth)", train_and_validate_randomforest(images, labels, test_images, test_labels, 10, 4))
# use Random forest with setting of trees = 10 and depth = 16
print("Untouched data - Random Forest (10 Trees, 16 Depth)", train_and_validate_randomforest(images, labels, test_images, test_labels, 10, 16))
# use Random forest with setting of trees = 30 and depth = 4
print("Untouched data - Random Forest (30 Trees, 4 Depth)", train_and_validate_randomforest(images, labels, test_images, test_labels, 30, 4))
# use Random forest with setting of trees = 30 and depth = 16
print("Untouched data - Random Forest (30 Trees, 16 Depth)", train_and_validate_randomforest(images, labels, test_images, test_labels, 30, 16))

# RANDOM FOREST - STRETCHED DATA
# use Random forest with setting of trees = 10 and depth = 4
print("Untouched data - Random Forest (10 Trees, 4 Depth)", train_and_validate_randomforest(stretched_image, labels, stretched_test_image, test_labels, 10, 4))
# use Random forest with setting of trees = 10 and depth = 16
print("Untouched data - Random Forest (10 Trees, 16 Depth)", train_and_validate_randomforest(stretched_image, labels, stretched_test_image, test_labels, 10, 16))
# use Random forest with setting of trees = 30 and depth = 4
print("Untouched data - Random Forest (30 Trees, 4 Depth)", train_and_validate_randomforest(stretched_image, labels, stretched_test_image, test_labels, 30, 4))
# use Random forest with setting of trees = 30 and depth = 16
print("Untouched data - Random Forest (30 Trees, 16 Depth)", train_and_validate_randomforest(stretched_image, labels, stretched_test_image, test_labels, 30, 16))

'''
one_digit = images[39009].reshape(28, 28)
label = labels[39009]
plt.title('Label is {label}'.format(label=label))
plt.imshow(one_digit, cmap='gray')
plt.show()

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
'''