# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 22:34:27 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

Weights=tf.Variable(tf.random_uniform([1],-1,1))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step %20 ==0:
        print(step,sess.run(Weights),sess.run(biases))


###########placeholder
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

##########定义网络层
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise   


                
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)


hg=plt.figure()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if(i%50==0):
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

#########################minist分类
# from tensorflow.examples.tutorials.mnist import input_data
# mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#
# xs = tf.placeholder(tf.float32,[None,784])
# ys = tf.placeholder(tf.float32,[None,10])
# prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#
# train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# sess=tf.Session()
#
#
# sess.run(tf.global_variables_initializer())
#
# batch_xs,batch_ys=mnist.train.next_batch(100)
# sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#
# def compute_accuracy(v_xs,v_ys):
#     global prediction
#     y_pre=sess.run(prediction,feed_dict={xs:v_xs})
#     correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#     result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
#     return result
# for i in range(1000):
#     sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#     if (i%50==0):
#         print(compute_accuracy(mnist.test.images,mnist.test.lables)
#
#
# ################dropout
# from sklearn.datasets import load_digits
# from sklearn.cross_validation import train_test_split
# from sklearn.preprocessing import LabelBinarizer
#
# keep_prob=tf.placeholder(tf.float32)
#
#


























    
