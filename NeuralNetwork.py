import numpy
import scipy.special    # sigmoid函数


# 神经网络类
class NeuralNetwork:
    """简单三层神经网络类

    输入层、隐藏层和输出层
    """
    def __init__(self, input_num, hidden_num, output_num):
        self.i_num = input_num
        self.h_num = hidden_num
        self.o_num = output_num
        # 权重矩阵
        # 正态分布
        self.wih = numpy.random.normal(0.0, pow(self.h_num, -0.5), (self.h_num, self.i_num))
        self.who = numpy.random.normal(0.0, pow(self.o_num, -0.5), (self.o_num, self.h_num))

        # 激活函数
        self.activation_func = lambda x: scipy.special.expit(x)
        self.inverse_activation_func = lambda x: scipy.special.logit(x)
        pass

    # 训练
    def train(self, inputs_list, targets_list, lr):
        # 转换成数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隐藏层
        hidden_outputs = self.activation_func(numpy.dot(self.wih, inputs))

        # 输出层
        final_outputs = self.activation_func(numpy.dot(self.who, hidden_outputs))

        # 误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新权重
        self.who += lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T)
        self.wih += lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)
        pass

    # 查询
    def query(self, inputs):
        # 输入层->隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        # 隐藏层->输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        return final_outputs

    # 反向查询
    def back_query(self, targets):
        final_outputs = numpy.array(targets, ndmin=2).T
        final_inputs = self.inverse_activation_func(final_outputs)

        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs = hidden_outputs * 0.98 + 0.01
        hidden_inputs = self.inverse_activation_func(hidden_outputs)

        inputs = numpy.dot(self.wih.T, hidden_inputs)
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs = inputs * 0.98 + 0.01

        return inputs

    def get_model(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        获得模型矩阵

        :return: 模型的两个权重矩阵wih和who
        """
        return self.wih, self.who

    def set_model(self, wih: numpy.ndarray, who: numpy.ndarray):
        """
        设置模型权重矩阵

        :param wih: 矩阵wih
        :param who: 矩阵who
        """
        self.wih = wih
        self.who = who
        pass
    pass
