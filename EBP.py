
import random 

import csv 
import math 
 

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 

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
 

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])



def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    for i in range(n_hidden):
        hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} ]
    network.append(hidden_layer)
    for i in range(n_outputs):
        output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} ]
    network.append(output_layer)
    return network
 

def evaluate_algorithm(dataset, algorithm, l_rate,n_epoch , n_hidden):

    Y_train=[]
    X_train=[]
    Y_test=[]
    X_test=[]
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
        X_train.append(i[0:7])
        Y_train.append(i[7])

    for i in test:
        X_test.append(i[0:7])
        Y_test.append(i[7])


    predicted = algorithm(train, test, l_rate,n_epoch , n_hidden)
    actual = Y_test
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def sigmoid(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    p=0
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
                p=p+error
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
                p=p+(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid(neuron['output'])
    return(p)

        

            
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

            
            
def train_network(network, train, l_rate, n_epoch, n_outputs):
    q=0
    avg=0
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            q=q+backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        avg=q/len(train)
        print(avg)

            


def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)
 
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 

# load and prepare data
filename = '/Users/abhibhagupta/Desktop/computer_networks/seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)


l_rate = 0.3
n_epoch = 5
n_hidden = 1

scores = evaluate_algorithm(dataset, back_propagation, l_rate,n_epoch , n_hidden)
print('Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


