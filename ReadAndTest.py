import json
import numpy
import NeuralNetwork


model_dir = r'./Model'
model_name = '100_1_1.txt'
testing_file_loc = r'./MnistDataSets/mnist_test.csv'

with open(model_dir + '/' + model_name, 'r') as f:
    json_data = json.load(f)
    input_node_num = json_data['input_num']
    hidden_node_num = json_data['hidden_num']
    output_node_num = json_data['output_num']
    wih = numpy.array(json_data['wih'])
    who = numpy.array(json_data['who'])
    pass

n = NeuralNetwork.NeuralNetwork(input_node_num, hidden_node_num, output_node_num)
n.set_model(wih, who)

with open(testing_file_loc, 'r') as tf:
    test_data_list = tf.readlines()

    score = 0.0
    for test_data in test_data_list:
        all_values = test_data.split(',')
        correct_num = int(all_values[0])
        outputs = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        output_num = numpy.argmax(outputs)
        if output_num == correct_num:
            score += 1.0
        else:
            print("输出错误：将", correct_num, "识别为", output_num)
            pass
        pass
    print("分数：", score / len(test_data_list))
    pass

