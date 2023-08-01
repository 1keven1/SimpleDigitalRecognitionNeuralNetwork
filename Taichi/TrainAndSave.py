import numpy as np
import taichi as ti
import os
import json
import time

import NeuralNetworkTi

training_file_loc = r'../MnistDataSets/mnist_train.csv'
save_dir = r'../Model'
save_name = '100_1_1.txt'
input_node_num = 784
hidden_node_num = 300
output_node_num = 10
learning_rate = 0.1
epochs = 1


def show_progress(percentage):
    print('\r', round(percentage * 100, 2), "%", end='')
    pass


ti.init(arch=ti.cuda, default_fp=ti.f64)

# 加载训练集
training_data_file = open(training_file_loc, 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

n = NeuralNetworkTi.NeuralNetworkTi(input_node_num, hidden_node_num, output_node_num, epochs)

# 训练
print('训练开始')
start_time = time.time()

for i in range(len(training_data_list)):
    all_values = training_data_list[i].split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_node_num) + 0.01
    targets[int(all_values[0])] = 0.99
    np_inputs = np.array(inputs, ndmin=2).T
    np_targets = np.array(targets, ndmin=2).T
    n.train(np_inputs, np_targets, learning_rate)

    show_progress(i / len(training_data_list))
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
        'wih': wih.to_numpy().tolist(),
        'who': who.to_numpy().tolist()
    }
    json.dump(save_data, f)
    pass
