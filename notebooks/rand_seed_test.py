# Trying to understand the behavior of tensorflow random seeds
# JPC 2017/09/01
#
# Note: ipynb produced entirely unpredictable behavior (seemingly all permutations of seed behavior),
# so I am putting this in a script to remove one more potential source.

# ### Some links: 
#- https://github.com/tensorflow/tensorflow/issues/9171
#- https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/xOBGT-Zawd0
#- https://www.tensorflow.org/api_docs/python/tf/set_random_seed is the most thorough description, though the behavior is not always consistent with its implications.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

###
print('\n-----------------------------------------------')
print('-----------------------------------------------')

#tf.set_random_seed(0)
x1 = tf.random_normal([1], seed=0)
x2 = tf.random_normal([1])

print ('Example one: x1 op seed set, x2 not set, graph-level not set')
with tf.Session() as sess0:
    print('x1: {0} ; x2: {1}'.format(sess0.run(x1),sess0.run(x2)))
with tf.Session() as sess1:
    print('x1: {0} ; x2: {1}'.format(sess1.run(x1),sess1.run(x2)))
print('Docs say x1 should repeat, x2 should not.')


###
print('-----------------------------------------------')

tf.set_random_seed(0)
x1 = tf.random_normal([1])
x2 = tf.random_normal([1])

print ('Example two: x1 op seed not set, x2 not set, graph-level set')
with tf.Session() as sess0:
    print('x1: {0} ; x2: {1}'.format(sess0.run(x1),sess0.run(x2)))
with tf.Session() as sess1:
    print('x1: {0} ; x2: {1}'.format(sess1.run(x1),sess1.run(x2)))
print('Docs say x1 should repeat, x2 should also (https://www.tensorflow.org/api_docs/python/tf/set_random_seed).')

###
print('-----------------------------------------------')

tf.set_random_seed(0)
x1 = tf.random_normal([1], seed=0)
x2 = tf.random_normal([1], seed=0)

print ('Example three: x1 op seed set, x2 set, graph-level set')
with tf.Session() as sess0:
    print('x1: {0} ; x2: {1}'.format(sess0.run(x1),sess0.run(x2)))
with tf.Session() as sess1:
    print('x1: {0} ; x2: {1}'.format(sess1.run(x1),sess1.run(x2)))
print('Docs vaguely note: If both the graph-level and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.')
print('Regardless that behavior seems wrong...')

print('-----------------------------------------------')
print('-----------------------------------------------\n')


print('Same test but now with tf.Variable()')

###
print('\n-----------------------------------------------')
print('-----------------------------------------------')
#tf.set_random_seed(0)
x1 = tf.Variable(tf.random_normal([1], seed=0))
x2 = tf.Variable(tf.random_normal([1]))
print ('Example one: x1 op seed set, x2 not set, graph-level not set')
with tf.Session() as sess6:
    sess6.run(x1.initializer) 
    sess6.run(x2.initializer) 
    print('x1: {0} ; x2: {1}'.format(sess6.run(x1),sess6.run(x2)))

with tf.Session() as sess7:
    sess7.run(x1.initializer)
    sess7.run(x2.initializer) 
    print('x1: {0} ; x2: {1}'.format(sess7.run(x1),sess7.run(x2)))
print('Docs say x1 should repeat, x2 should not.  With tf.Variable, this behavior seems opposite?!')

print('-----------------------------------------------')
tf.set_random_seed(0)
x1 = tf.Variable(tf.random_normal([1]))
x2 = tf.Variable(tf.random_normal([1]))
print ('Example two: x1 op seed not set, x2 not set, graph-level set')
with tf.Session() as sess8:
    sess8.run(x1.initializer) 
    sess8.run(x2.initializer) 
    print('x1: {0} ; x2: {1}'.format(sess8.run(x1),sess8.run(x2)))
with tf.Session() as sess9:
    sess9.run(x1.initializer) 
    sess9.run(x2.initializer) 
    print('x1: {0} ; x2: {1}'.format(sess9.run(x1),sess9.run(x2)))
print('Docs say x1 should repeat, x2 should also (https://www.tensorflow.org/api_docs/python/tf/set_random_seed).')

###
print('-----------------------------------------------')

tf.set_random_seed(0)
x1 = tf.random_normal([1], seed=0)
x2 = tf.random_normal([1], seed=0)

print ('Example three: x1 op seed set, x2 set, graph-level set')
with tf.Session() as sess10:
    print('x1: {0} ; x2: {1}'.format(sess10.run(x1),sess10.run(x2)))
with tf.Session() as sess11:
    print('x1: {0} ; x2: {1}'.format(sess11.run(x1),sess11.run(x2)))
print('Docs vaguely note: If both the graph-level and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.')
print('Regardless that behavior seems wrong...')
print('\n\n')

