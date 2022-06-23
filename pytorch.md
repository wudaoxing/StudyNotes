```python
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F

torch.linspace(-3,3,100000)
# 在（-3，3）区间内划分出100000个点

torch.rand()
# 生成在[0,1)区间内均匀分布的随机数

torch.randn()
# 创建服从标准正态分布的一组随机数Tensor

torch.ones()
# 创建元素全为1的Tensor

torch.zeros()
# 创建元素全为0的Tensor

torch.max()
# 返回指定维度的最大值和序列号

torch.normal(4*x,2)
# 生成4为期望值，2为标准差的一批随机数据

torch.cat()
# 沿某维度方向进行连接操作，返回Tensor

torch.unsqueeze()
# 将Tensor的形状中指定维度添加维度1

torch.cuda.is_available()
# 对运行平台进行CUDA检测

nn.Linear()
# 用nn中预设好的线性的神经网络模型来构造线性模型
# 第一个参数代表输入数据的维度，第二个参数代表输出数据的维度

nn.MSELoss()
# nn模块中预设有均方误差函数，可以直接作为损失函数

nn.CrossEntropyLoss()
# 交叉熵损失函数

F.relu()
# PyTorch中预编写的ReLU激活函数

optim.SGD()
# PyTorch预设的随机梯度下降函数
# 第一个参数是需要优化的神经网络模型的参数，第二个参数是学习率

forward()
# 用来构造神经网络前向传播时的计算步骤

plt.cla()
# 清空图像画布

plt.scatter()
# 绘制散点图形式的数据样本点

plt.plot()
# 绘制回归直线

plt.text()
# 打印loss值

zero_()
# 清空梯度值
```
