import numpy
import NeuralNetwork
import os
import json
import time
import scipy


def show_progress(percentage):
    print('\r', round(percentage * 100, 2), "%", end='')
    pass


training_file_loc = r'./MnistDataSets/mnist_train.csv'
save_dir = r'./Model'
save_name = '100_1_1_Rot.txt'
input_node_num = 784
hidden_node_num = 100
output_node_num = 10
learning_rate = 0.1
epochs = 1

n = NeuralNetwork.NeuralNetwork(input_node_num, hidden_node_num, output_node_num)

# 加载训练集csv文件
training_data_file = open(training_file_loc, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练
print('训练开始')
start_time = time.time()
for e in range(epochs):
    for i in range(len(training_data_list)):
        all_values = training_data_list[i].split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_node_num) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets, learning_rate)

        # 旋转正负10度
        input_rotate_image_1 = scipy.ndimage.rotate(inputs.reshape(28, 28), 10, cval=0.01, reshape=False)
        input_rotate_image_2 = scipy.ndimage.rotate(inputs.reshape(28, 28), -10, cval=0.01, reshape=False)
        n.train(input_rotate_image_1.reshape(784), targets, learning_rate)
        n.train(input_rotate_image_2.reshape(784), targets, learning_rate)

        # 进度
        show_progress((i + len(training_data_list) * e) / (epochs * len(training_data_list)))
        pass
    pass
time_span = time.time() - start_time
print('\n训练完成，耗时', format(time_span, '.2f'), 's')

# 保存到文件
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    pass
with open(save_dir + '/' + save_name, 'w') as f:
    wih, who = n.get_model()
    save_data = {
        'input_num': input_node_num,
        'hidden_num': hidden_node_num,
        'output_num': output_node_num,
        'wih': wih.tolist(),
        'who': who.tolist()
    }
    json.dump(save_data, f)
    pass
