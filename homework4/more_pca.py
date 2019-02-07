# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt


def unpickle(file_name):
    file = open(file_name, 'rb')
    unpickled = pickle.load(file, encoding='latin1')
    file.close()
    return unpickled


def scatter_plot_error(error_vec):
    _, axis = plt.subplots()
    axis.scatter(np.arange(10), error_vec)

    for i, txt in enumerate(labels):
        axis.annotate(txt, (np.arange(10)[i], error_vec[i]))

    plt.title("Sum-Squared Error")
    plt.show()

def scatter_plot_PCA(data, title):
    _, axis = plt.subplots()
    axis.scatter(data.T[:, 0], data.T[:, 1])
    plt.title(title)

    for i, txt in enumerate(labels):
        axis.annotate(txt, (data.T[i, 0], data.T[i, 1]))
    plt.show()

def principal_coordinate_analysis(dist):
    A_matrix = np.identity(10) - np.reshape(np.ones(100) / 10, (10, 10))

    W_matrix = -0.5 * np.dot(np.dot(A_matrix.T, dist), A_matrix)

    eigen_vals, eigen_vect = np.linalg.eig(W_matrix)

    idx = eigen_vals.argsort()[::-1]
    eigen_vect = eigen_vect[:, idx]
    l = np.dot(np.dot(eigen_vect.T, W_matrix), eigen_vect)

    #get the number of dimensions we want to represent and get their sqrt
    sigma = l[0:2, 0:2]
    sigma[0, 0] = np.sqrt(sigma[0, 0])
    sigma[1, 1] = np.sqrt(sigma[1, 1])

    Y = np.dot(sigma, eigen_vect[:, 0:2].T)

    return Y


def calculate_error_per_item(pca3, label_i, cov_mat_i):
    Xi = np.dot(pca3.transform(label_i), pca3.components_)
    Xi = np.add(Xi, cov_mat_i)
    error = np.sum(np.square(Xi - label_i))

    return error


def probability_similarity_measure(category, category_means, a_b_dir):
    error_matrix = np.zeros((10, 10))
    for i in range(9):
        pca3 = PCA(n_components=20)
        if a_b_dir is True:
            pca3.fit(category[i+1])
        else:
            pca3.fit(category[i])
        for j in range(i + 1, 10):
            if a_b_dir is True:
                label_i = category[j]
                cov_mat_i = category_means[i, :]
            else:
                label_i = category[i]
                cov_mat_i = category_means[j, :]
            error_matrix[i, j] = calculate_error_per_item(pca3, label_i, cov_mat_i)
            error_matrix[j, i] = error_matrix[i, j]  # make the matrix symetric

    return error_matrix


def load_data(data):
    features = data[0]['data']
    classes = data[0]['labels']
    for i in range(1, len(data)):
        features = np.concatenate((features, data[i]['data']))
        classes = np.concatenate((classes, data[i]['labels']))

    return features, classes


def data_reader():
    pickle_data = ["cifar-10-batches-py/data_batch_1", "cifar-10-batches-py/data_batch_2", "cifar-10-batches-py/data_batch_3",
                   "cifar-10-batches-py/data_batch_4", "cifar-10-batches-py/data_batch_5", "cifar-10-batches-py/test_batch"]

    data = []

    for i in range(len(pickle_data)):
        data.append(unpickle(pickle_data[i]))

    features, classes = load_data(data)
    features = features / 255  # rescale X
    all_data = np.column_stack((features, classes))

    return all_data, features, classes


def construct_mean_category():
    dt, features, classes = data_reader()

    mean_images = pd.DataFrame(dt[:, 0:3072]).groupby(pd.Series(dt[:, 3072]), axis=0).mean()
    categories = []

    for i in range(10):
        categories.append(features[classes == i])

    mean_images = np.asarray(mean_images)

    return mean_images, categories


def calculate_euclidean_distance(mean_images):
    initial_distance = np.zeros((10, 10))

    for i in range(10):
        for j in range(i + 1, 10):
            initial_distance[i, j] = np.linalg.norm((mean_images[i, :] - mean_images[j, :]))
            initial_distance[j, i] = initial_distance[i, j]

    return initial_distance


############################
# Part A: Get Error for each image
#############################


def estimate_squared_error(categories, category_means, n_classes, n_pcas):

    list_x = []
    error = []
    error_unscaled = []
    twentieth_pca = []

    for i in range(n_classes):
        pca = PCA()
        pca.fit(categories[i])
        twentieth_pca.append(pca.components_[:n_pcas, :])
        print(twentieth_pca)
        class_pcas = pca.transform(categories[i])[:, :n_pcas]
        components = pca.components_[:n_pcas, :]
        list_x.append(np.dot(class_pcas, components))
        list_x[i] = np.add(list_x[i], category_means[i, :])
        print(list_x)
        error.append(np.sum(np.square(list_x[i] - categories[i])))
        error_unscaled.append(np.sum(np.square(255 * (list_x[i] - categories[i]))))

    return np.asarray(error)


labels = ["airplain", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
category_means, categories = construct_mean_category()

###################################
# Plot the squared error
###################################

error_array = estimate_squared_error(categories, category_means, 10, 20)
scatter_plot_error(error_array)


###################################
# Part B
###################################


dist = calculate_euclidean_distance(category_means)
print(dist)
np.savetxt("partb_distances1.csv", dist, delimiter=",", fmt='%5.4f')

#######################################

#2D scatter plot(Euclidean distance)

#######################################
V = principal_coordinate_analysis(dist)
scatter_plot_PCA(V, "2D scatter plot(Euclidean distance)")


#######################################

#Part C: (1/2)(E(A → B) + E(B → A))

#######################################
error_a_b = probability_similarity_measure(categories, category_means, True)
error_b_a = probability_similarity_measure(categories, category_means, False)

error_total = 0.5 * (error_a_b + error_b_a)
print(error_total)


np.savetxt("partc_distances1.csv", error_total, delimiter=",", fmt='%5.4f')

#######################################

#2D scatter plot(Probability similarity metric)

#######################################
V2 = principal_coordinate_analysis(error_total)
scatter_plot_PCA(V2, "2D scatter plot(Probability similarity metric)")
