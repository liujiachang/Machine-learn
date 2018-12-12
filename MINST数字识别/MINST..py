'''
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

labels_images = pd.read_csv('train.csv',header=None,sep=',', low_memory=False)
images = np.array(labels_images.iloc[1:, 1:])
labels = np.array(labels_images.iloc[1:, :1])
images = np.array(images, dtype=int)
#print(type(images[0][0]))
for i in range(len(images)):
    for j in range(len(images[i])):
        if images[i][j]>0:
            images[i][j]=1
labels = labels.flatten()
labels = np.array(labels, dtype=int)
x_train, x_test, y_train, y_test = train_test_split(images[:5000], labels[:5000], test_size=0.2)

#print(x_train.shape,y_train.shape)
## knn k=3 socre = 0.927
## svm c=4 gamma=0.01 socre = 0.959
#for i in range(1,10):
clf = svm.SVC(C=4,gamma=0.01)
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))

test = pd.read_csv('test.csv')
test = np.array(test)
print(len(test))
for i in range(len(test)):
    for j in range(len(test[i])):
        if test[i][j]>0:
            test[i][j]=1
results = clf.predict(test)
print(len(results))
a = [i for i in range(1,len(test)+1)]
df = pd.DataFrame({'ImageId':a,'Label':results})
df.to_csv('result.csv',index=False,sep=',')
'''

#读入数据

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import random
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = np.array(train.values.tolist())
test = np.array(test.values.tolist())
#print(len(test))
images = np.array(train[:, 1:])
labels = np.array(train[:, 0])
# 编码及标准化
onehot_encoder = OneHotEncoder(sparse=False)
min_max_scaler = MinMaxScaler()
labels = labels.reshape(len(images),1)
labels = onehot_encoder.fit_transform(labels)
images = min_max_scaler.fit_transform(images)
test = min_max_scaler.fit_transform(test)
#print(images, labels)

sess = tf.InteractiveSession()
#定义占位符x和y_
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
#定义用于初始化的两个函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#定义卷积和池化的函数
#卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入大小相同
#池化用简单传统的2x2大小的模板做max pooling，因此输出的长宽会变为输入的一半
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
     return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#第一层卷积，卷积在每个5x5的patch中算出32个特征
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
#第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积，每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#有1024个神经元的全连接层，此时图片大小为7*7
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，在输出层之前加入dropout。用一个placeholder代表一个神经元的输出在dropout中保持不变的概率。
#这样可以在训练过程中启用dropout，在测试过程中关闭dropout。
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#应为 y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#训练和评估模型
#用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.argmax(y_conv,1)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):  #为减少训练时间，降低迭代次数
    random.seed(i)
    x_batch = random.sample(list(images), 50)
    random.seed(i)
    y_batch = random.sample(list(labels), 50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
#print(prediction.eval(feed_dict={x: list(test[:50]), keep_prob: 1.0}))
testy = []
for i in range(560):
    result = prediction.eval(feed_dict={x: list(test[i*50:(i+1)*50]), keep_prob: 1.0})
    testy.extend(result)
#print(len(testy))
#print(testy)
a = [i for i in range(1, 28001)]
df = pd.DataFrame({'ImageId': a, 'Label': testy})
df.to_csv('result.csv', index=False, sep=',')