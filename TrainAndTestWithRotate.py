import numpy
import NeuralNetwork
import scipy


training_file_loc = r'./MnistDataSets/mnist_train.csv'
testing_file_loc = r'./MnistDataSets/mnist_test.csv'
input_node_num = 784
hidden_node_num = 150
output_node_num = 10
learning_rate = 0.05
epochs = 2

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

        input_rotate_image_1 = scipy.ndimage.rotate(inputs.reshape(28, 28), 10, cval=0.01, reshape=False)
        input_rotate_image_2 = scipy.ndimage.rotate(inputs.reshape(28, 28), -10, cval=0.01, reshape=False)
        n.train(input_rotate_image_1.reshape(784), targets, learning_rate)
        n.train(input_rotate_image_2.reshape(784), targets, learning_rate)
        pass
    pass

# 测试
test_data_file = open(testing_file_loc, 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score = 0
for test_data in test_data_list:
    all_values = test_data.split(',')
    correct_num = int(all_values[0])
    outputs = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    output_num = numpy.argmax(outputs)
    if output_num == correct_num:
        score += 1
    else:
        print("输出错误：将", correct_num, "识别为", output_num)
        pass
    pass
print("分数：", score)
