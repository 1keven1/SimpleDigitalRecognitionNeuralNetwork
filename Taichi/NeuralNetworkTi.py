import taichi as ti
import taichi.math as tm
import numpy as np


@ti.func
def activation_func(x):
    return 1.0 / (1.0 + tm.exp(-x))


@ti.func
def inverse_activation_func(x):
    return tm.log(x / (1 - x))


@ti.data_oriented
class NeuralNetworkTi:
    def __init__(self, input_num, hidden_hum, output_num, epochs=1):
        """
        神经网络类

        :param input_num: 输入节点数量
        :param hidden_hum: 隐藏节点数量
        :param output_num: 输出节点数量
        """
        self.i_num = input_num
        self.h_num = hidden_hum
        self.o_num = output_num
        self.epochs = epochs

        # 权重矩阵
        np_wih = np.random.normal(0.0, pow(hidden_hum, -0.5), (hidden_hum, input_num))
        np_who = np.random.normal(0.0, pow(output_num, -0.5), (output_num, hidden_hum))
        self.wih = ti.field(ti.f64, (self.h_num, self.i_num))
        self.who = ti.field(ti.f64, (self.o_num, self.h_num))
        self.wih.from_numpy(np_wih)
        self.who.from_numpy(np_who)

        # 初始化各种矩阵
        self.inputs = ti.field(ti.f64, (self.i_num, 1))
        self.targets = ti.field(ti.f64, (self.o_num, 1))
        self.hidden_outputs = ti.field(ti.f64, (self.h_num, 1))
        self.final_outputs = ti.field(ti.f64, (self.o_num, 1))
        self.final_errors = ti.field(ti.f64, (self.o_num, 1))
        self.hidden_errors = ti.field(ti.f64, (self.h_num, 1))
        pass

    @ti.kernel
    def train(self, input_list: ti.types.ndarray(), target_list: ti.types.ndarray(), lr: ti.f64):
        # 复制数据到field
        for i, j in input_list:
            self.inputs[i, j] = input_list[i, j]
            pass

        for i, j in target_list:
            self.targets[i, j] = target_list[i, j]
            pass

        # 清空矩阵
        self.hidden_outputs.fill(0)
        self.final_outputs.fill(0)
        self.final_errors.fill(0)
        self.hidden_errors.fill(0)
        # 计算
        # hidden_outputs = activation_func(dot(wih, inputs))
        for i, j in self.hidden_outputs:
            for m in range(self.i_num):
                self.hidden_outputs[i, j] += self.wih[i, m] * self.inputs[m, j]
                pass
            self.hidden_outputs[i, j] = activation_func(self.hidden_outputs[i, j])
            pass

        # final_outputs = activation_func(dot(who, hidden_outputs))
        for i, j in self.final_outputs:
            for m in range(self.h_num):
                self.final_outputs[i, j] += self.who[i, m] * self.hidden_outputs[m, j]
                pass
            self.final_outputs[i, j] = activation_func(self.final_outputs[i, j])
            pass

        # 误差
        # final_errors = targets - final_outputs
        for i, j in self.final_errors:
            self.final_errors[i, j] = self.targets[i, j] - self.final_outputs[i, j]
            pass

        # hidden_errors = dot(who.T, final_errors)
        for i, j in self.hidden_errors:
            for m in range(self.o_num):
                self.hidden_errors[i, j] += self.who[m, i] * self.final_errors[m, j]
                pass
            pass

        # 更新权重
        # who += lr * dot(final_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
        for i, j in self.who:
            gradient = self.final_errors[i, 0] * self.final_outputs[i, 0] * (1 - self.final_outputs[i, 0]) * \
                       self.hidden_outputs[0, j]
            self.who[i, j] += lr * gradient
            pass

        # wih += lr * dot(hidden_error * hidden_outputs * (1 - hidden_outputs), inputs.T)
        for i, j in self.wih:
            gradient = self.hidden_errors[i, 0] * self.hidden_outputs[i, 0] * (1 - self.hidden_outputs[i, 0]) * \
                       self.inputs[0, j]
            self.wih[i, j] += lr * gradient
            pass
        pass

    # 查询
    @ti.kernel
    def query(self, input_list: ti.types.ndarray()):
        # 复制数据到field
        for i, j in input_list:
            self.inputs[i, j] = input_list[i, j]
            pass

        # 输入层->隐藏层
        # hidden_outputs = activation_func(dot(wih, inputs))
        for i, j in self.hidden_outputs:
            for m in range(self.i_num):
                self.hidden_outputs[i, j] += self.wih[i, m] * self.inputs[m, j]
                pass
            self.hidden_outputs[i, j] = activation_func(self.hidden_outputs[i, j])
            pass

        # final_outputs = activation_func(dot(who, hidden_outputs))
        for i, j in self.final_outputs:
            for m in range(self.h_num):
                self.final_outputs[i, j] += self.who[i, m] * self.hidden_outputs[m, j]
                pass
            self.final_outputs[i, j] = activation_func(self.final_outputs[i, j])
            pass
        pass

    def get_output(self):
        return self.final_outputs

    def get_model(self) -> tuple[ti.field, ti.field]:
        return self.wih, self.who

    def set_model(self, wih, who):
        self.wih = wih
        self.who = who
        pass

    pass
