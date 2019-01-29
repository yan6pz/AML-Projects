import numpy as np
import csv
import matplotlib.pyplot as plt


class supportVectorMachine:
    def __init__(self, weight, b=0.0, reg_lambda=1e-1, step_length=0.1):
        self.reg_lambda = reg_lambda
        self.weight = np.array(weight)
        self.b = b
        self.X = np.array([])
        self.Y = np.array([])
        self.cost = 0
        self.training_cost = np.array([])
        self.step_length = step_length

    def cost_function(self, X, Y):
        self.training_cost = 1 - (np.dot(X, self.weight)+self.b)*Y
        self.training_cost[self.training_cost<0] = 0
        self.cost = np.mean(self.training_cost) + self.reg_lambda*np.dot(self.weight.T, self.weight)/2
        print(" training cost is ", self.cost)

    def update_weight(self):
        #print("current weight: ", self.weight)
        weight_to_update=self.X*self.Y.reshape((self.Y.shape[0],1))
        #print("X is:", self.X)
        #print("Y is:", self.Y)
        #print("weight_to_update of yx", weight_to_update)
        zero_cost_matrix = 1 - (np.dot(self.X, self.weight)+self.b)*self.Y
        zero_cost_matrix[zero_cost_matrix<0]=0
        #print("zero cost maxtrix as filter: ", zero_cost_matrix)
        weight_to_update = weight_to_update*((zero_cost_matrix!=0).reshape((zero_cost_matrix.shape[0],1)))
        #print("masking 'cost = 0', weight_to_update", weight_to_update)
        #print("regularization term is ", self.reg_lambda*self.weight)
        weight_to_update = -(1/self.X.shape[0])*np.sum(weight_to_update,axis=0)+self.reg_lambda*self.weight
        #print("final weight_to_update: ",weight_to_update)
        self.weight = self.weight - self.step_length*weight_to_update
        #print("new weight: ", self.weight)
        # to update b
        #print("current b: ", self.b)
        self.b = self.b - self.step_length*(-np.dot(self.Y, 1*(zero_cost_matrix!=0))/self.X.shape[0])
        #print("updated_b: ", self.b)

    def StochasticGradientDesc(self, X, Y):
        self.X = X
        self.Y = Y
        self.update_weight()

    def predict(self, X):
        #print("raw output is: ", np.dot(X, self.weight)+self.b)
        return 2*((np.dot(X, self.weight)+self.b)>0)-1

    def set_learningRate(self, lr):
        self.step_length = lr


data = []
X = []
Y = []
# import the data from the csv.
with open('./homework2/train.txt', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        data.append(row)

data = np.array(data)
np.random.shuffle(data)
# extract only continuous variable values to form X
X = data[:, (0,2,4,10,11,12)].astype(float)
# extract last col to form classes of 1 for >50K and -1 for <=50K
Y = 2*(data[:, 14] == ' >50K')-1
# rescale the features to same variance and zero means.
X = (X - np.mean(X, axis=0))/np.std(X,axis=0)
rescaled_data = np.column_stack((X,Y))

split_idx = int(data.shape[0]*0.1)
hyperParmSearch_data = rescaled_data[:split_idx, :]
train_data = rescaled_data[split_idx:, :]

#regularisation_lambda = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1]
regularisation_lambda = [1e-3, 1e-2, 1e-1, 1]
step_length_m = 1
step_length_n = 300
total_epoch = 60
steps = 300
batch_size = 1
accuracy_history = np.zeros((len(regularisation_lambda), int(steps*total_epoch/30)))
svm = None
max_achieved_accuracy = 0
max_achieved_weight = []
for idx_lambda in range(len(regularisation_lambda)):
    weight = np.random.rand(X.shape[1])
    svm = supportVectorMachine(weight=weight, reg_lambda=regularisation_lambda[idx_lambda])
    for i in range(total_epoch):
        print("******epoch: ", i," *******")
        lr = step_length_m/(0.01*i+step_length_n)
        svm.set_learningRate(lr)

        np.random.shuffle(train_data)
        held_out = train_data[:50, :]
        train = train_data[50:, :]
        for j in range(1, steps+1):
            selected = np.random.randint(train.shape[0], size=batch_size)
            svm.StochasticGradientDesc(train[selected, :-1], train[selected, -1])
            if j % 30 == 0:
                print("--->Step: ", j, " <----")
                validation_result = svm.predict(hyperParmSearch_data[:, :-1])
                validation_accuracy = sum(validation_result == hyperParmSearch_data[:, -1]) / hyperParmSearch_data.shape[0]
                print(" Validation accuracy is ", validation_accuracy*100, "%")
                accuracy_history[idx_lambda, int((i*steps+j)/30)-1]=validation_accuracy
                if validation_accuracy >= max_achieved_accuracy:
                    max_achieved_accuracy = validation_accuracy
                    max_achieved_weight = svm.weight
                #svm.cost_function(hyperParmSearch_data[:, :-1], hyperParmSearch_data[:, -1])
            '''
                validation_result = svm.predict(held_out[:, :-1])
                validation_accuracy = sum(validation_result == held_out[:, -1]) / held_out.shape[0]
                print(" Validation accuracy is ", validation_accuracy*100, "%")
                svm.cost_function(held_out[:, :-1], held_out[:, -1])
            '''

'''
test_result = svm.predict(hyperParmSearch_data[:, :-1])
test_accuracy = sum(test_result == hyperParmSearch_data[:, -1]) / hyperParmSearch_data.shape[0]
print(" Test data accuracy is ", test_accuracy*100, "%")
svm.cost_function(hyperParmSearch_data[:, :-1], hyperParmSearch_data[:, -1])
'''
x_axis = range(int(steps*total_epoch/30))
plt.plot(x_axis, accuracy_history[0])
plt.plot(x_axis, accuracy_history[1])
plt.plot(x_axis, accuracy_history[2])
plt.plot(x_axis, accuracy_history[3])
plt.ylim((0.5,1))
plt.legend([regularisation_lambda[0], regularisation_lambda[1], regularisation_lambda[2], regularisation_lambda[3]], loc='lower right')
plt.show()

def save_for_submission(results):
    fobj = open('./homework2/submission.txt', 'a+')
    for i in results:
        if i >= 1:
            fobj.write('>50K\n')
        else:
            fobj.write('<=50K\n')
    fobj.close()


grader_data = []
with open('./homework2/test.txt', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        grader_data.append(row)

grader_data = np.array(grader_data)
grader_X = grader_data[:, (0,2,4,10,11,12)].astype(float)
# rescale the features to same variance and zero means.
grader_X = (grader_X - np.mean(grader_X, axis=0))/np.std(grader_X,axis=0)
grader_result = svm.predict(grader_X)
save_for_submission(grader_result)


