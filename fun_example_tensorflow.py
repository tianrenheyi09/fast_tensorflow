
######------------------  tf.concat
# import tensorflow as tf
# t1 = [[[1, 1, 1],[2, 2, 2]],[[3, 3, 3],[4, 4, 4]]]
# t2 = [[[5, 5, 5],[6, 6, 6]],[[7, 7, 7],[8, 8, 8]]]
# with tf.Session() as sess:
#     A = tf.concat([t1, t2],axis=0)
#     B = tf.concat([t1, t2],axis=1)
#     C = tf.concat([t1, t2],axis=2)
#     print(sess.run(A))
#     print('---------------------------------------------------------')
#     print(sess.run(B))
#     print('---------------------------------------------------------')
#     print(sess.run(C))

import tensorflow as tf
w1=tf.Variable(tf.random_uniform([2,3],-1,1),name="w1")
ww1 = tf.Variable(1,name="w1")
w2 = tf.get_variable('w1',shape=[3,3])
print(w1.name)
print(ww1.name)
print(w2.name)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))

