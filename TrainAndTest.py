import numpy
import NeuralNetwork


# 训练参数
training_file_loc = r'./MnistDataSets/mnist_train.csv'
testing_file_loc = r'./MnistDataSets/mnist_test.csv'
input_node_num = 784
hidden_node_num = 150
output_node_num = 10
learning_rate = 0.1
epochs = 2

# 实例化神经网络
n = NeuralNetwork.NeuralNetwork(input_node_num, hidden_node_num, output_node_num)

# 加载训练集csv文件
training_data_file = open(training_file_loc, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练
for e in range(epochs):
    for training_data in training_data_list:
        all_values = training_data.split(',')
        # 从第二个元素开始是输入值 将其从[0, 255]映射到[0.01, 0.99]
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 第一个元素是数字的值 预期值为其它0.01 对应的为0.99
        targets = numpy.zeros(output_node_num) + 0.01
        targets[int(all_values[0])] = 0.99
        # 千训练
        n.train(inputs, targets, learning_rate)
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
