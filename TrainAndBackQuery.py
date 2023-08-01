import numpy
import NeuralNetwork
import matplotlib.pyplot
import pylab


training_file_loc = r'./MnistDataSets/mnist_train.csv'
testing_file_loc = r'./MnistDataSets/mnist_test.csv'
input_node_num = 784
hidden_node_num = 150
output_node_num = 10
learning_rate = 0.1
epochs = 4

n = NeuralNetwork.NeuralNetwork(input_node_num, hidden_node_num, output_node_num)

# 加载训练集csv文件
training_data_file = open(training_file_loc, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# print(len(training_data_list))

# 训练
for e in range(epochs):
    for training_data in training_data_list:
        all_values = training_data.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_node_num) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets, learning_rate)
        pass
    pass

# 测试
lable = 3
targets = numpy.zeros(output_node_num) + 0.01
targets[lable] = 0.99
print(targets)

img_data = n.back_query(targets)
matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')
pylab.show()
