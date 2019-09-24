import csv
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
            row[column] = float(row[column].strip())
            

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


class Neuron:
    def __init__(self,n_attributes,learning_rate):
        self.weights=[]
        for i in range(n_attributes+1):
            self.weights.append(0)
            self.learning_rate=learning_rate
            
    def predict(self,inputs):
        weighted_sum=0
        for i in range(len(self.weights)):
            weighted_sum+=self.weights[i]*inputs[i]
        if weighted_sum>=0:
            return 1
        else:
            return -1
       
    def update_weights(self,inputs,target):
        prediction=self.predict(inputs)
        error=target-prediction
        del_weights=[]
        if (error):
            for i in range(len(self.weights)):
                del_weights.append(inputs[i]*self.learning_rate*error)
            for i in range(len(self.weights)):
                self.weights[i]=self.weights[i]+del_weights[i]


import csv
import random
filename = '/Users/abhibhagupta/Desktop/computer_networks/seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

for i in dataset:
    i.insert(0,1)
    
X_test=[]
Y_test=[]
X_train=[]
Y_train=[]
n_train=0
n_test=0
N=len(dataset)
n_train=int((N*90)/100)
n_test=N-n_train


train=[]
test=[]
test_points=set()
for i in range(n_test):
    n=random.randint(0,N)
    test_points.add(n)

matrix=[]
for i in range(N):
    matrix.append(0)
for i in test_points:
    test.append(dataset[i-1])
    matrix[i-1]=1
for i in range(len(matrix)):
    if matrix[i]==0:
        train.append(dataset[i-1])
        


for i in train:
    X_train.append(i[0:8])
    Y_train.append(i[8])


for i in test:
    X_test.append(i[0:8])
    Y_test.append(i[8])

attributes=7
learning_parameter=0.1
epochs=5000
N = Neuron(attributes,learning_parameter)

for j in range(epochs):
    for i in range(len(X_train)):
        N.update_weights(X_train[i],Y_train[i])
        
count=0

for i in range(len(Y_test)):
    Y_pred = N.predict(X_test[i])
    error = Y_test[i] - Y_pred
    if(error):
        count += 1
print('Accuracy:',100 -(count/len(X_test))*100)

