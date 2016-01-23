import NeuralNetwork
import pickle
#from random import shuffle
import random
import numpy as np
import externalFunctions


with open('./data_dump/objs.pickle_train_small') as f:
    dataset_data, dataset_labels = pickle.load(f)


with open('./data_dump/objs.pickle_test') as f:
    validation_data, validation_labels = pickle.load(f)

# Shuffle the dataset
# data_shuf = []
# labels_shuf = []
# index_shuf = range(len(dataset_labels))
# random.shuffle(index_shuf)
# for i in index_shuf:
#     data_shuf.append(dataset_data[i])
#     labels_shuf.append(dataset_labels[i])


# We will train over a smaller train dataset for finding hyperparameters
#train_data = dataset_data[0:1500]
#train_labels = dataset_labels[0:1500]
#validation_data = dataset_data[1500:]
#validation_labels = dataset_labels[1500:]

learning_rates = np.logspace(-3,1,10)
regularization_terms = np.logspace(-3, -1, 10)
number_epochs = range(1,11)
number_hidden_units = range(50,151)

number_trials = [10]#,20,30,40,50,100]

best_validation_errors = []
best_hyp_parameters_list = []


for trials in number_trials:
    validation_metrics = []
    hyp_parameters_list = []
    for i in range(trials):
        learning_rate = random.choice(learning_rates)
        regularization_term = random.choice(regularization_terms)
        number_epoch = random.choice(number_epochs)
        hidden_units = random.choice(number_hidden_units)

        # Set of hyperparameters
        hyp_parameters = [learning_rate, regularization_term,
                          number_epoch, hidden_units]

        # Train
        nn = NeuralNetwork.NeuralNetwork(150, hidden_units, 1)
        nn.SGDbackProp(dataset_data, dataset_labels,number_epoch,
                        learning_rate,regularization_term)
        #nn.SGDbackProp(train_data, train_labels,10,
        #               0.01,0.001)
                       #validation_data=validation_data,
                       #validation_labels=validation_labels)


        validation_metrics.append(externalFunctions.getPrecisionRecallSupport(nn, validation_data, validation_labels))
        hyp_parameters_list.append(hyp_parameters)



