import json
import numpy as np
import NeuralNetworkTi
import taichi as ti


model_dir = r'../Model'
model_name = '100_1_1.txt'
testing_file_loc = r'../MnistDataSets/mnist_test.csv'

ti.init(ti.cuda, default_fp=ti.f64)

with open(model_dir + '/' + model_name, 'r') as f:
    json_data = json.load(f)
    input_node_num = json_data['input_num']
    hidden_node_num = json_data['hidden_num']
    output_node_num = json_data['output_num']
    np_wih = np.array(json_data['wih'])
    np_who = np.array(json_data['who'])
    pass

wih = ti.field(ti.f64, (hidden_node_num, input_node_num))
who = ti.field(ti.f64, (output_node_num, hidden_node_num))
wih.from_numpy(np_wih)
who.from_numpy(np_who)

n = NeuralNetwork.NeuralNetworkTi(input_node_num, hidden_node_num, output_node_num)
n.set_model(wih, who)

with open(testing_file_loc, 'r') as tf:
    test_data_list = tf.readlines()

    score = 0.0
    for test_data in test_data_list:
        all_values = test_data.split(',')
        correct_num = int(all_values[0])
        n.query(np.array((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01, ndmin=2).T)
        outputs = n.get_output().to_numpy()
        output_num = np.argmax(outputs)
        if output_num == correct_num:
            score += 1.0
        else:
            # print("输出错误：将", correct_num, "识别为", output_num)
            pass
        pass
    print("分数：", score / len(test_data_list))
    pass

