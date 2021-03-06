## Load DVS data
import aerdathelper as ah
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CONSTANTS
DVSX = 128
DVSY = 128
DVSRES = DVSX * DVSY

## Load data
#file = 'M1a_expand.aedat'
#file = 'mipleft.aedat'
file = 'onight_6_6.aedat'
bytes2read = 0
ts, xs, ys, ps = ah.loadaerdat(datafile=file, length=bytes2read, version='aedat', debug=0, camera='DVS128')

# pre-process data
inds = [(100, 1200), 
        (2148268, 2150268), 
        (3582081 , 3583981),
        (5005769 , 5007969),
       (6440257 , 6442057),
       (7866669 , 7868769),
       (9299182 , 9301132),
       (10727995 , 10730045)]

past_accums = None
future_accums = None
for ind in inds:
    past, future = ah.dvs2accum(xs[ind[0]:ind[1]], ys[ind[0]:ind[1]], ts[ind[0]:ind[1]], 20)
    if past_accums is None:
        past_accums = past
        future_accums = future
    else:
        past_accums = np.concatenate((past_accums, past), axis=0)
        future_accums = np.concatenate((future_accums, future), axis=0)

print("Data processed, about to build graph")

num_dssizes = 8
ds_sizes = [2**x for x in range(num_dssizes)]
learning_rate = 0.01
batch_size = 128
test_size = 256
img_width = 128
img_height = 128
num_inp = img_width * img_height
num_outp = img_width * img_height
write_image = True

trX = np.reshape(past_accums, [-1, img_width * img_height])
trY = np.reshape(future_accums, [-1, img_width * img_height])
teX = trX
teY = trY

def pool_layer(inp, size):
    """ Return tensorflow OPs that will average patches of size size in the inp
        tensor. Results are scaled between 0,1.
    """
    pool_size = [1, size, size, 1]
    fourD = tf.reshape(inp, [-1, img_width, img_height, 1])
    avg = tf.nn.avg_pool(fourD, ksize=pool_size, strides=pool_size, padding='VALID')
    # Rescale as: normalize_value = (value − min_value) / (max_value − min_value)
    return tf.divide(
            tf.subtract(
                avg, 
                tf.reduce_min(avg)),
            tf.subtract(
                tf.reduce_max(avg),
                tf.reduce_min(avg))
        )

with tf.Graph().as_default() as g:
    X = tf.placeholder(tf.float32, shape=[None, num_inp], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, num_outp], name='Y')
    
    W = tf.Variable(tf.random_normal(shape=[num_inp, num_outp], stddev=0.1), name='W')
    b = tf.Variable(tf.random_normal(shape=[num_outp], stddev=0.01), name='b')

    y = tf.nn.relu(tf.add(tf.matmul(X, W), b), name='outp')

    ## Calculate the cost of resolutions 0 (128 wide), 3 (16 wide), 5 (4 wide)
    c3_past = pool_layer(y, ds_sizes[3])
    c5_past = pool_layer(y, ds_sizes[5])
    
    c3_future = pool_layer(Y, ds_sizes[3])
    c5_future = pool_layer(Y, ds_sizes[5])
    
    c0_rms = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(y, Y), 2)))
    c3_rms = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(c3_past, c3_future), 2)))
    c5_rms = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(c5_past, c5_future), 2)))
    
    cost = tf.add(c0_rms, tf.add(c3_rms, c5_rms), name='total_cost')

    #cost = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(y, Y), 2, name='powerrr')))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

error = []
print('starting tf session')
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(3):
        
        training_batch = zip(range(0, len(trX), batch_size),
                        range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

        error.append(sess.run(cost, feed_dict={X: teX[test_indices], Y: teY[test_indices]}))
        print(i, error[-1])
    test_indices = np.arange(len(teX)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]
    pred = sess.run(y, feed_dict={X: teX[test_indices], Y: teY[test_indices]})



    ## If on my laptop then just write straight to images
    if write_image:
        print("Attempting to write images now...")
        import visTools
        image_dir = 'imgs/'
        kx = DVSX
        visTools.write_preds(teX[test_indices], teY[test_indices], pred, image_dir, kx)



"""       
    # Plot the prediction
    plt.matshow(pred[1].reshape(img_width, img_height))

# Analysis
plt.matshow(pred[0].reshape(img_width, img_height))

plt.figure()
plt.plot(error)
plt.title('RMSE during training')
"""
