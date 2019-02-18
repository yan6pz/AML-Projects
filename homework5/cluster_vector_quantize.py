import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

activity = ["Brush_teeth", "Climb_stairs", "Comb_hair", "Descend_stairs",
            "Drink_glass", "Eat_meat", "Eat_soup", "Getup_bed",
            "Liedown_bed", "Pour_water", "Sitdown_chair", "Standup_chair",
            "Use_telephone", "Walk"]

class Tree:
    def __init__(self, data=None):
        self.children = []
        self.data = data
    def add_child(self, node):
        self.children.append(node)
    def printTree(self):
        print('node data is', self.data)
        for each in self.children:
            each.printTree()



class HierarchicalKmean:
    def __init__(self, structure, sample_size=0.4):
        self.level = len(structure)
        self.structure = structure
        self.sample_size = sample_size
        self.kmeans_tree = Tree()

    def fit(self,input):
        samples = np.array(input)[np.random.choice(len(input), int(self.sample_size*len(input)))]
        self.kmeans_tree.data=KMeans(n_clusters=self.structure[0]).fit(samples)
        results = self.kmeans_tree.data.predict(input)
        for each_cluster in range(self.structure[0]):
            data_in_cluster = input[results==each_cluster]
            newNode = Tree(KMeans(n_clusters=self.structure[1]).fit(data_in_cluster))
            self.kmeans_tree.add_child(newNode)
        self.kmeans_tree.printTree()
        return self

    def predict(self, input):
        return np.array([self.predict_one(each) for each in input])

    def predict_one(self, input):
        input = np.array(input).reshape((1,-1))
        if (self.kmeans_tree.data):
            intrimResult = self.kmeans_tree.data.predict(input)
            return self.kmeans_tree.children[intrimResult[0]].data.predict(input)[0]+intrimResult[0]*self.structure[1]
        else:
            print("fit your Hierarchical Kmean model to data first")


def split_sequence(data, piece_length, step_length):
    vectors = []
    for one_sample in data:
        end_idx = 0
        # following will split into (N-piece_length)/step_length + 1
        while (end_idx+piece_length) <= len(one_sample[0]):
            vectors.append(one_sample[0][end_idx:(end_idx+piece_length)])
            end_idx = end_idx+step_length

        #further take the remaining data (if any) to form one final piece
        if ((end_idx) < len(one_sample[0])-1):
            vectors.append(one_sample[0][-piece_length:])
    vectors = np.array(vectors)
    vectors = vectors.reshape((vectors.shape[0],-1),order='F')
    return vectors


def compute_clusters(vectors, k_cluster):
    kmeans = KMeans(n_clusters=k_cluster).fit(vectors)
    #kmeans = HierarchicalKmean([40,12]).fit(vectors)
    return kmeans


def plot_histogram(bins, cluster_labels):
    plt.hist(cluster_labels, bins)
    plt.title("Class Histogram")
    plt.show()

def vector_quantize_build_dictionary(data, piece_length, step_length, k_cluster):
    pieces_vectors=split_sequence(data, piece_length, step_length)
    kmeans = compute_clusters(pieces_vectors, k_cluster)
    return kmeans


def vector_quantize_represent_signal(one_sample, piece_length, step_length, kmeans):
    one_sample = one_sample.reshape((1,-1))
    pieces_vector = split_sequence(one_sample, piece_length, step_length)
    results = kmeans.predict(pieces_vector)
    # compute the histogram vector and normalize it by total count
    histogram = [sum(results==each) for each in range(k_cluster)]
    return np.array(histogram)/len(results)


def quantize_all_data(data, piece_length, step_length, kmeans):
    vectorized_data = [np.concatenate([vector_quantize_represent_signal(one_sample, piece_length, step_length, kmeans),[one_sample[1]]]) for one_sample in data]
    return np.array(vectorized_data)


def train_and_validate_randomforest(train_data, train_labels, test_data, test_label, n_trees, depth):
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
    clf.fit(train_data, train_labels)
    predicted = clf.predict(test_data)
    accuracy = sum(predicted == test_label)/test_label.shape[0]
    return accuracy


def split_training_test_set(input_data, folds):
    test_set = []
    train_set = []
    for idx, each_class in enumerate(activity):
        current_activity = input_data[input_data[:,-1]==idx]
        X_train, X_test, y_train, y_test = train_test_split(current_activity[:,:-1], current_activity[:,-1], test_size=1/folds)
        train_set.append(np.column_stack((X_train, y_train)))
        test_set.append(np.column_stack((X_test, y_test)))
    train_set = np.concatenate(train_set)
    np.random.shuffle(train_set)
    test_set = np.concatenate(test_set)
    return train_set, test_set


data = []
for i, _ in enumerate(activity):
    files = os.listdir("./homework5/HMP_Dataset/"+activity[i])
    for file in files:
        sequence_data = []
        fobj = open("./homework5/HMP_Dataset/"+activity[i]+"/"+file, "r")
        for line in fobj:
            fields = line.split()
            sequence_data.append(fields)
        data.append(np.array([np.array(sequence_data).astype(float),i]))
data = np.array(data)

piece_length = 32 # 32hz per second, so we take just 1 second of data into a piece.
step_length = 10
k_cluster = 4

kmeans = vector_quantize_build_dictionary(data,piece_length, step_length, k_cluster)

vectorized_data = quantize_all_data(data, piece_length, step_length, kmeans)

train_st, test_st = split_training_test_set(vectorized_data,3)

accuracy = train_and_validate_randomforest(train_st[:,:-1], train_st[:,-1], test_st[:,:-1], test_st[:,-1], 30, 16)