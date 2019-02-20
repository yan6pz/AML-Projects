import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
        #self.kmeans_tree.printTree()
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
        if (end_idx) < len(one_sample[0])-1:
            vectors.append(one_sample[0][-piece_length:])
    vectors = np.array(vectors)
    vectors = vectors.reshape((vectors.shape[0],-1),order='F')
    return vectors


def compute_clusters(vectors, k_cluster, hierarchical=False, hierarch_structure = None):
    if hierarchical and hierarch_structure is not None:
        kmeans = HierarchicalKmean(hierarch_structure).fit(vectors)
    else:
        kmeans = KMeans(n_clusters=k_cluster).fit(vectors)
    return kmeans

def vector_quantize_build_dictionary(data, piece_length, step_length, k_cluster, hierarchical=False, hierarch_structure = None):
    pieces_vectors=split_sequence(data, piece_length, step_length)
    kmeans = compute_clusters(pieces_vectors, k_cluster, hierarchical, hierarch_structure)
    return kmeans

def vector_quantize_represent_signal(one_sample, piece_length, step_length, kmeans, k_cluster):
    one_sample = one_sample.reshape((1,-1))
    pieces_vector = split_sequence(one_sample, piece_length, step_length)
    results = kmeans.predict(pieces_vector)
    # compute the histogram vector and normalize it by total count
    histogram = [sum(results==each) for each in range(k_cluster)]
    return np.array(histogram)/len(results)

def quantize_all_data(data, piece_length, step_length, kmeans, k_cluster):
    vectorized_data = [np.concatenate([vector_quantize_represent_signal(one_sample, piece_length, step_length, kmeans, k_cluster),[one_sample[1]]]) for one_sample in data]
    return np.array(vectorized_data)


def plot_histogram(plt, vectorized_data, activity_id):
    bins = vectorized_data.shape[1]-1  # reduce by 1 which is the label.
    vectors = vectorized_data[vectorized_data[:,-1] == activity_id][:,:-1]
    mean_vector = np.mean(vectors, axis=0)
    plt.bar(np.arange(bins), mean_vector)
    plt.title("Class Histogram for "+activity[activity_id])
    plt.show()

def train_and_validate_randomforest(train_data, train_labels, test_data, test_label, n_trees, depth):
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
    clf.fit(train_data, train_labels)
    predicted = clf.predict(test_data)
    accuracy = sum(predicted == test_label)/test_label.shape[0]
    return accuracy, predicted


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

accuracy_list = []
best_accuracy = 0
kmean_list = None
piece_length_list = [32, 48]
k_cluster_list = [100, 110, 120]
#hierarch_struct = [[40,12],[40, 15], [40,20]]
#hierarch_struct = [[10,3],[10, 4], [10,5], [10, 6], [10, 7]]
#hierarch_struct = [[10,4],[16, 10], [40,6], [40, 12]]
#hierarch_struct = [[40,6]]

#piece_length = 32 # 32hz per second, so we take just 1 second of data into a piece.

skip_portion = 0.3  # overlap% will be 1-skip_portion

for piece_length in piece_length_list:
    step_length = int(piece_length*skip_portion)
    for idx, k_cluster in enumerate(k_cluster_list):
        kmeans = vector_quantize_build_dictionary(data,piece_length, step_length, k_cluster)
        #kmeans = vector_quantize_build_dictionary(data, piece_length, step_length, k_cluster, hierarchical=True, hierarch_structure=hierarch_struct[idx])
        vectorized_data = quantize_all_data(data, piece_length, step_length, kmeans, k_cluster)
        repeats = 3
        average_accuracy = 0
        for iterate in range(repeats):
            train_st, test_st = split_training_test_set(vectorized_data,3)
            accuracy, predicted = train_and_validate_randomforest(train_st[:,:-1], train_st[:,-1], test_st[:,:-1], test_st[:,-1], 120, 30)
            #print('iteration #',iterate," accuracy is: ", accuracy)
            average_accuracy += accuracy
            if accuracy >= best_accuracy:
                kmean_list=kmeans
                print(confusion_matrix(test_st[:, -1], predicted))
                print('Best accuracy is: ', accuracy)
                best_accuracy = accuracy
        average_accuracy /= repeats
        print('piece_length:', piece_length, ' K:', k_cluster, ' Average Accuracy is: ', average_accuracy)
        accuracy_list.append([piece_length, k_cluster, average_accuracy])



#fobj = open("./homework5/accuracy_trees80_depth32.csv",'a+')
fobj = open("./homework5/accuracy.csv",'a+')
for row in np.array(accuracy_list):
    #fobj.write(', '.join(row.astype(str)) + ',Hierarchical-Kmean,'+str((1-skip_portion)*100)+'%\n')
    fobj.write(', '.join(row.astype(str)) + ',standard,'+str((1-skip_portion)*100)+'%\n')
fobj.close()


def plot_all_histogram(vectorized_data):
    for idx, _ in enumerate(activity):
        plt.subplot(4, 4, idx+1)
        plot_histogram(plt, vectorized_data, idx)
    plt.subplots_adjust(hspace=0.6)
    plt.show()

def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots()
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            c = matrix[i][j]
            ax.text(i + 0.5, j + 0.5, str(c), va='center', ha='center')
    ax.set_xlim(0, len(matrix))
    ax.set_ylim(0, len(matrix))
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_yticks(np.arange(len(matrix)))
    ax.grid()

plot_all_histogram(vectorized_data)
