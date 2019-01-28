import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt

np.random.seed(12)


# Calculation of gamma i (aT*xi+b)
def calc_gamma_i(x, a, b):
    a = np.array(a)
    result = (np.dot(a, x) + b)
    return result


#updating model parameters following the gradient decent procedure
def gradient_descent(feature_x, yi, matrix_A, b, l, epoch, stochastic_step_len, stochastic_step_size):
    gamma = calc_gamma_i(feature_x[0], matrix_A, b)
    # Calculate batch size
    batch_size = 1 / ((stochastic_step_len * epoch) + stochastic_step_size)

    if yi * gamma >= 1:
        matrix_A = matrix_A - (batch_size * l * matrix_A)
    else :
        penalty = yi * feature_x
        reg_params = np.array(l) * matrix_A
        error_coef = reg_params - penalty
        matrix_A = np.array([i - j for i, j in zip(matrix_A, batch_size * error_coef)])[0][0]  
        b = b - (batch_size * -yi)
        
    return matrix_A, b


def write_array_to_file(name, list):
    with open(name+'.txt', 'w') as f:
        for item in list:
            f.write("%s\n" % item)


def accuracy(x, y, A, b):
    predictions = predict(x, len(y), A, b)
    correct = 0

    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            correct += 1
          
    return correct / float(len(y))


# Get the magnitude of the coefficents vector
def normalize_vector(x):
    return math.sqrt(sum(x**2))


def plot_images(vector, l, ylabel, ylim):
    plt.plot(vector,'r')
    plt.title("Lambda = {}".format(l))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim([0, 50])
    plt.show()


def train(l, epochs, steps):
    matrix_a = [0.0 for w in range(np.shape(train_X)[1])]
    b = 0
    accuracy_vector = []
    magnitude_vector = []

    for epoch in range(epochs):
        # Select 50 examples for each epoch
        index= np.random.randint(len(train_X), size=50)
        accuracy_data = train_X[index]
        accuracy_labels = train_Y[index]
        train_data = train_X[-index]
        train_labels = train_Y[-index]
        # For each of the steps in an epoch
        for step in range(steps):
            row_index = np.random.randint(len(train_labels), size=1)
            feature_x = train_data[row_index]
            yi= np.asmatrix(train_labels[row_index])
      
            # Estimate gamma and update the parameters
            matrix_a, b = gradient_descent(feature_x, yi, matrix_a, b, l, epoch, stochastic_step_len, stochastic_step_size)

            # Get magnitude and validation accuracy every 30 steps
            if step % 30 == 0:
                accuracy_val = accuracy(accuracy_data, accuracy_labels, matrix_a, b)
                accuracy_vector.append(accuracy_val)
                magnitude=normalize_vector(matrix_a)
                magnitude_vector.append(magnitude)

    plot_images(accuracy_vector, l, 'Penalty error', [0, 1])
    plot_images(magnitude_vector, l, 'Magnitude', [0, 3])
    
    val_acc = accuracy(validate_X, validate_Y, matrix_a, b)
        
    return val_acc, matrix_a, b


def predict(x, n_len, A, b):
    predictions = []
    for i in range(n_len):
        row = x[i, :]
        gamma = calc_gamma_i(row, A, b)
        if gamma >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    return predictions


names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]



def train_test_validate_split(features, labels):
    #split 90:10 training:validation
    train_x, validation_x, train_y, validatation_y = train_test_split(features, labels, test_size=0.1,
                                                                  random_state=12)
    
    return train_x, train_y,  validation_x, validatation_y


def encode_labels(labels_y):
    labels = []
    for row in list(labels_y):
        row = str(row).strip().replace('.', "")

        if row == "<=50K":
            labels.append(-1)
        else:
            labels.append(1)
    return pd.Series(labels)


def decode_labels(labels_y):
    labels = []
    for row in list(labels_y):
        if row == -1:
            labels.append("<=50K")
        else:
            labels.append(">50K")
    return labels


def process_training_data():
    train_data = pd.read_csv('train.csv', names=names, na_values=" ?")

    train_data = train_data.iloc[:, [0,2,4,10,11,12,14]]

    data = train_data.dropna()
    data_X = data.iloc[:, 0:6]
    data_clean_X = data_X.loc[:, data_X.dtypes == "int64"]
    data_clean_Y = data.iloc[:, -1]
    X_scaled = preprocessing.scale(data_clean_X)
    encoded_Y = encode_labels(data_clean_Y)

    return X_scaled,encoded_Y


def process_testing_data():
    test_data = pd.read_csv('test.csv', names=names, na_values=" ?")

    test_data = test_data.iloc[:, [0, 2, 4, 10, 11, 12]]

    data = test_data.dropna()
    data_X = data.iloc[:, 0:6]
    data_clean_X = data_X.loc[:, data_X.dtypes == "int64"]
    X_scaled = preprocessing.scale(data_clean_X)

    return X_scaled


X_scaled, encoded_Y = process_training_data()
train_X, train_Y, validate_X, validate_Y = train_test_validate_split(X_scaled, encoded_Y)

train_X = np.array(train_X)
train_Y = np.array(train_Y)
validate_X = np.array(validate_X)
validate_Y = np.array(validate_Y)

stochastic_step_len = .01
stochastic_step_size = 50
lambdas = [1e-3, 1e-2, 1e-1, 1e-0]


########################################################
# Train and validate SVM on validation dataset
########################################################
validation_accuracies = {}
for l in lambdas:
    validation_accuracies[l] = train(l, 50, 300)
        
print(validation_accuracies)


########################################################
# Predict using SVM's trained parameters on testing dataset
########################################################
x_test = process_testing_data()
test_predictions = {}
for l in lambdas:
    _, matrix_A, b = validation_accuracies[l]
    test_predictions[l] = predict(np.array(x_test), len(x_test), matrix_A, b)
    decoded_pred = decode_labels(test_predictions[l])
    write_array_to_file("Lambda = {}".format(l), decoded_pred)



