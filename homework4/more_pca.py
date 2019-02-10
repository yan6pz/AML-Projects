import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pickle
import math
import matplotlib.pyplot as plt


def unpickle(file_name):
    file = open(file_name, 'rb')
    unpickled = pickle.load(file, encoding='bytes')
    file.close()
    return unpickled


def load_data(data):
    features = data[0][b'data']
    classes = data[0][b'labels']
    for i in range(1, len(data)):
        features = np.concatenate((features, data[i][b'data']))
        classes = np.concatenate((classes, data[i][b'labels']))
    return features, classes


def bar_plot_error(error_vec, labels):
    plt.bar(np.arange(len(error_vec)), error_vec)
    plt.title("Sum-Squared Error")
    plt.xticks(np.arange(len(error_vec)), labels)
    plt.show()

def scatter_plot_PCA(data, title, labels):
    _, axis = plt.subplots()
    axis.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    for i, txt in enumerate(labels):
        axis.annotate(txt, (data[i, 0], data[i, 1]))
    plt.show()

def construct_category_img_w_error(all_data):
    category_mean = []
    category_comps = []
    category_recon_img = []
    category_error = []
    for i in range(10):
        this_category = all_data[all_data[:, -1] == i][:,:-1]
        category_mean.append(np.mean(this_category, axis=0))
        #pca = PCA(n_components=20, svd_solver='full')
        pca = PCA(n_components=20)
        pca.fit(this_category)
        category_comps.append(pca.components_)
        reconstructed = pca.inverse_transform(pca.transform(this_category))
        category_recon_img.append(reconstructed)
        category_error.append(np.sum((reconstructed - this_category)**2)/reconstructed.shape[0])
    return category_mean, category_comps, category_recon_img, category_error


def calculate_euclidean_distance(mean_images):
    fobj = open('./homework4/partb_distances.csv', 'a+')
    distance = squareform(pdist(mean_images, 'euclidean'))**2
    for row in distance:
        fobj.write(', '.join(row.astype(str)) + '\n')
    fobj.close()
    return distance

def principal_coordinate_analysis(pairDistance):
    A = np.eye(pairDistance.shape[0]) - np.ones(pairDistance.shape)/pairDistance.shape[0]
    W = np.dot(np.dot(A,pairDistance),A.T)/(-2.0)
    eig_value, eig_vector = np.linalg.eig(W)
    reorder = np.argsort(eig_value)[::-1]
    eig_value = eig_value[reorder]
    eig_vector = eig_vector[:, reorder]
    s = 2
    sigma_s = np.sqrt(eig_value[0:s])*np.eye(s)
    low_d_points = np.dot(eig_vector[:,0:2], sigma_s)
    scatter_plot_PCA(low_d_points, "multi-dimensional scaling",labels)

def calculate_E_A_B_error(A_category, B_category):
    part_c_pca = PCA(n_components=20)
    part_c_pca.fit(B_category)
    part_c_pca.mean_= np.mean(A_category, axis=0)
    reconstructed_A_B = part_c_pca.inverse_transform(part_c_pca.transform(A_category))
    return np.sum((reconstructed_A_B - A_category)**2)/reconstructed_A_B.shape[0]

def calculate_EAB_distance(all_data):
    D = np.zeros((10,10))
    for i in range(10):
        for j in range(i,10):
            A_category = all_data[all_data[:, -1] == i][:,:-1]
            B_category = all_data[all_data[:, -1] == j][:,:-1]
            D[i,j] = (calculate_E_A_B_error(A_category, B_category)+calculate_E_A_B_error(B_category, A_category))/2
            if i!=j:
                D[j,i] = D[i,j]
    fobj = open("./homework4/partc_distances.csv",'a+')
    for row in D:
        fobj.write(', '.join(row.astype(str)) + '\n')
    fobj.close()
    return D


pickle_data = ["./homework4/cifar-10-batches-py/data_batch_1", "./homework4/cifar-10-batches-py/data_batch_2", "./homework4/cifar-10-batches-py/data_batch_3",
                   "./homework4/cifar-10-batches-py/data_batch_4", "./homework4/cifar-10-batches-py/data_batch_5", "./homework4/cifar-10-batches-py/test_batch"]

data = []
for i in range(len(pickle_data)):
    data.append(unpickle(pickle_data[i]))

features, classes = load_data(data)
all_data = np.column_stack((features, classes))

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

############################
# Part A: Get Error for each image
#############################
category_mean, category_comps, category_recon_img, category_error = construct_category_img_w_error(all_data)
# Plot the squared error
bar_plot_error(category_error,labels)

###################################
# Part B
###################################
pairDistance = calculate_euclidean_distance(category_mean)
principal_coordinate_analysis(pairDistance)

#######################################
#Part C: (1/2)(E(A → B) + E(B → A))
#######################################
D = calculate_EAB_distance(all_data)
principal_coordinate_analysis(D)


'''mean_centered = all_data[all_data[:, -1] == 0][:,:-1]-category_mean[0]
converted = np.dot(part_c_pca.components_, mean_centered.T)
restored_A_B = np.dot(part_c_pca.components_.T, converted).T + category_mean[0]'''

'''def plot_mean_image(category_mean, labels):
    fig, axes1 = plt.subplots(2, 5, figsize=(5, 5))
    for j in range(2):
        for k in range(5):
            axes1[j][k].set_axis_off()
            axes1[j][k].set_title(labels[j*5+k])
            axes1[j][k].imshow(category_mean[j*5+k].astype(int).reshape((32,32,3), order='F').transpose((1,0,2)))
    plt.show()


plot_mean_image(category_mean, labels)

def estimate_squared_error(categories, category_means, n_classes, n_pcas):

    list_x = []
    error = []
    error_unscaled = []
    twentieth_pca = []

    for i in range(n_classes):
        pca = PCA()
        pca.fit(categories[i])
        twentieth_pca.append(pca.components_[:n_pcas, :])

        class_pcas = pca.transform(categories[i])[:, :n_pcas]
        components = pca.components_[:n_pcas, :]
        list_x.append(np.dot(class_pcas, components))
        list_x[i] = np.add(list_x[i], category_means[i, :])

        error.append(np.sum(np.square(list_x[i] - categories[i])))
        error_unscaled.append(np.sum(np.square(255 * (list_x[i] - categories[i]))))

    return np.asarray(error)

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

'''