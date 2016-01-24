import NeuralNetwork
import pickle



with open('./data_dump/objs.pickle_train') as f:
    dataset_data, dataset_labels = pickle.load(f)

# Train the whole network with fixed parameters and different values of each of them independently
learning_rates = [0.001, 0.01, 0.1, 1, 2]

nn = NeuralNetwork.NeuralNetwork(150,100,1)

for learning_rate in learning_rates:

    nn.SGDbackProp(dataset_data, dataset_labels,5, learning_rate)

    with open('./output/objs.pickle_nn_5_lr_{}'.format(learning_rate), 'w') as f:
        pickle.dump(nn, f)