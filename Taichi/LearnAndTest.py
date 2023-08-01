import numpy
import numpy as np
import taichi as ti

import NeuralNetworkTi


training_file_loc = r'../MnistDataSets/mnist_train.csv'
testing_file_loc = r'../MnistDataSets/mnist_test.csv'
input_node_num = 784
hidden_node_num = 100
output_node_num = 10
learning_rate = 0.1
epochs = 1

ti.init(arch=ti.gpu, default_fp=ti.f64)

# 加载训练集
training_data_file = open(training_file_loc, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

n = NeuralNetwork.NeuralNetworkTi(input_node_num, hidden_node_num, output_node_num, epochs)

# 训练
for training_data in training_data_list:
    all_values = training_data.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_node_num) + 0.01
    targets[int(all_values[0])] = 0.99
    np_inputs = np.array(inputs, ndmin=2).T
    np_targets = np.array(targets, ndmin=2).T
    n.train(np_inputs, np_targets, learning_rate)
    pass

# 测试
test_data_file = open(testing_file_loc, 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score = 0
for test_data in test_data_list:
    all_values = test_data.split(',')
    correct_num = int(all_values[0])
    n.query(np.array((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01, ndmin=2).T)
    outputs = n.get_output().to_numpy()
    output_num = numpy.argmax(outputs)
    if output_num == correct_num:
        score += 1
    else:
        print("输出错误：将", correct_num, "识别为", output_num)
        pass
    pass
print("分数：", score)

