import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.cluster.vq import vq
import os

np.random.seed(1234)


def extract_segments(data, seg_size):
    rows_size = np.shape(data)[0]
    #if there is multiplier that expands seg_size to data size grab all data
    if rows_size % seg_size != 0:
        vector_to_segment = np.array(data)[:-(rows_size % seg_size), :]
    else:
        vector_to_segment = np.array(data)

    number_of_segments = int(rows_size / seg_size)
    num_of_dimensions = seg_size * 3
    vector_segment = vector_to_segment.reshape(number_of_segments, num_of_dimensions)

    return pd.DataFrame(vector_segment)

def segment_vectors(dir, file, seg_size):
    path_to_file = os.path.join(dir, file)
    data = pd.read_csv(path_to_file, sep=" ", index_col=None, names=['x', 'y', 'z'], skip_blank_lines=True).dropna()
    segmented_data = extract_segments(data, seg_size)

    return segmented_data

def train_test_split(dir, train_split_ratio):
    all_files_per_dir = os.listdir(dir)
    np.random.shuffle(all_files_per_dir)
    train_count = int(train_split_ratio * len(all_files_per_dir))
    test_count = int((1 - train_split_ratio) * len(all_files_per_dir))

    return all_files_per_dir[:train_count], all_files_per_dir[-test_count:]

def train_test_split_per_file(dir, seg_size, train_split_ratio):

    train_data, test_data = train_test_split(dir, train_split_ratio)
    training_dataset = pd.DataFrame()
    testing_dataset = pd.DataFrame()

    for file in train_data:
        training_segmented_data = segment_vectors(dir, file, seg_size)
        training_dataset = training_dataset.append(training_segmented_data, ignore_index=True)

    for file in test_data:
        testing_segmented_data = segment_vectors(dir, file, seg_size)
        testing_dataset = testing_dataset.append(testing_segmented_data, ignore_index=True)

    return training_dataset, testing_dataset

def segment_feature_data(seg_size, train_split_ratio):
    training_dataset = []
    for activity in activities:
        training_set, _ = train_test_split_per_file(data_path + activity, seg_size, train_split_ratio)
        training_dataset.append(training_set)

    print("Data segmentation completed")

    return np.concatenate(training_dataset)

def generate_histogram_vector(model, n_cluster, seg_size, dir, file):
    path_to_file = os.path.join(dir, file)
    data_from_file = pd.read_csv(path_to_file, sep=" ", index_col=None, names=['x', 'y', 'z'], skip_blank_lines=True).dropna()
    segments = extract_segments(data_from_file, seg_size)

    #assign segments to clusters based on current's model cluster centers
    quantized_vector = np.array(vq(segments, model.cluster_centers_)[0])
    histogram_data = [0] * (n_cluster +1)

    for i in quantized_vector:
        histogram_data[i] += 1

    histogram_vector = histogram_data
    histogram_data[n_cluster] = activities.index(dir.split('/')[1]) + 1

    feature = pd.DataFrame(np.array(histogram_data).reshape(1, n_cluster + 1))

    return feature, histogram_vector

def train_test_split_classifier(model, dir, clusters_count, seg_size, train_split_ratio):
    train_data, test_data = train_test_split(dir, train_split_ratio)
    feature_data_train = pd.DataFrame()
    feature_data_test = pd.DataFrame()
    histogram_vectors = []

    for file in train_data:
        feature, histogram_vector = generate_histogram_vector(model, clusters_count, seg_size, dir, file)
        feature_data_train = feature_data_train.append(feature)
        histogram_vectors.append(histogram_vector)

    for file in test_data:
        feature,_ = generate_histogram_vector(model, clusters_count, seg_size, dir, file)
        feature_data_test = feature_data_test.append(feature)

    return feature_data_train, feature_data_test, np.mean(np.array(histogram_vectors), axis=0)


def plot_histograms(histogram_mean_vector , activity):
    plt.bar(np.arange(len(histogram_mean_vector)), histogram_mean_vector)
    plt.title("mean quantized vector: " + activity)
    plt.show()

def construct_features(feature_vector, clusters_count, seg_size, train_split_ratio, has_plot_histograms):
    k_means_model = KMeans(n_clusters=clusters_count, random_state=0).fit(feature_vector)
    print("Fitted K-means clustering model")
    # Generate training and test features for all the adl activity
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for activity in activities:
        training_set, testing_set, histogram_mean_vector = train_test_split_classifier(k_means_model, data_path + activity, clusters_count
                                                                           , seg_size, train_split_ratio)
        if has_plot_histograms:
            plot_histograms(histogram_mean_vector, activity)

        train_set = train_set.append(training_set)
        test_set = test_set.append(testing_set)
    print("Classifier features generated")
    return train_set, test_set

def predict(clusters_count, seg_size, train_split_ratio = 0.67, has_plot_histograms = False):
    print("Number Clusters= {} and segment length = {}".format(clusters_count, seg_size))
    feature_vector = segment_feature_data(seg_size, train_split_ratio)
    train_classifier, test_classifier = construct_features(feature_vector, clusters_count, seg_size
                                                                    , train_split_ratio, has_plot_histograms)

    random_forest = RandomForestClassifier(max_depth=32, random_state=0, n_estimators=80)

    random_forest.fit(train_classifier.iloc[:, :clusters_count], train_classifier.iloc[:, clusters_count])

    prediction = random_forest.predict(test_classifier.iloc[:, :clusters_count])

    source_of_truth = test_classifier.iloc[:, clusters_count]
    accuracy = accuracy_score(source_of_truth, prediction)
    print("Accuracy=" + str(accuracy * 100) + "%")
    print("Error rate= "+ str((1-accuracy)*100)+"%")
    print(confusion_matrix(source_of_truth, prediction))

    return accuracy

activities = ["Brush_teeth","Climb_stairs","Comb_hair","Descend_stairs","Drink_glass","Eat_meat","Eat_soup","Getup_bed","Liedown_bed","Pour_water",
                                                              "Sitdown_chair","Standup_chair","Use_telephone","Walk"]

data_path = "HMP_Dataset/"

####################################################################################################################
# Initial test
####################################################################################################################

n_centroids = 480
segment_length = 32

predict(n_centroids, segment_length, 0.67)

####################################################################################################################
# Best accuracy
####################################################################################################################

n_centroids = 25
segment_length = 4

predict(n_centroids, segment_length, 0.67, True)

####################################################################################################################
# Experiments for various segment length and number of clusters
####################################################################################################################

clusters_no = [25, 50, 200, 400]
seg_lengths = [4, 8, 32]

result = []
for seg_length in seg_lengths:
    interim = []
    for clusters in clusters_no:
        acc = predict(clusters, seg_length, 0.67)
        interim.append(acc*100)
    result.append(interim)


for i, y in enumerate(result):
    plt.plot(clusters_no, y, label=seg_lengths[i])

plt.xlabel("Clusters Number")
plt.ylabel("Accuracy")
plt.title("Accuracy variation for different seg size and # of clusters")
plt.ylim([0, 100])
plt.yticks(np.arange(0, 100, 5))
plt.legend()
plt.grid(True)
plt.savefig('accuracy_per_seg_size.png')
plt.close()

