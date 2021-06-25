# coding: utf-8
import sys, os
sys.path.append(os.getcwd())  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from four_layer_net import FourLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = FourLayerNet(input_size=784, hidden1_size=100, hidden2_size=100, hidden3_size=50, output_size=10)

#超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#学习中所有训练数据都被使用过一次时的更新次数
iter_per_epoch = max(train_size / batch_size, 1)    #60000/100

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 梯度
    grad = network.gradient(x_batch, t_batch)
    
    # 更新权重
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #每学习一个epoch，计算准确度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


#取10个进行验证
mask = np.arange(0,10)
x_batch = x_test[mask]
t_batch = t_test[mask]
y_batch = network.predict(x_batch)
for i in np.arange(10):
    print("predict=" + str(np.argmax(y_batch[i])) + ", label=" + str(np.argmax(t_batch[i])))

