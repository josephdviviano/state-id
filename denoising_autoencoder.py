#!/usr/bin/env python

from datetime import datetime
import numpy as np
import tables
import tensorflow as tf

#import matplotlib
#matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
#import matplotlib.pyplot as plt

class DenoisingAutoencoder:

    def __init__(self):
        # hidden nodes
        self.hid = [3000, 1000, 300, 10]
        self.depth = len(self.hid)
        self.current_depth = 1

        # encoded data (hdf5)
        self.encoded_data = None

        # learning rate
        self.lr = 0.00001
        self.batch_size = 100
        self.epoch = 10
        self.weightstd = 0.01
        self.debug = False
        self.weights = []
        self.biases = []
        self.training_loss = []


    def corrupt(self, X, pct=0.2):
        """masks data by setting indices to 0 in each sample"""
        n, m = X.shape
        X_corrupt = np.copy(X)

        # corrupt different features in each sample
        for i in range(n):
            idx = np.random.choice(m, int(m*pct), replace=False)
            X_corrupt[i, idx] = 0

        return X_corrupt


    def clean(self, X):
        """remove nans from data, if found"""
        idx = np.where(np.isnan(X))[0]
        if len(idx) > 0:
            print('removed {} nans'.format(len(idx)))
            X[np.isnan(X)] = 0
        return(X)


    def get_batches(self, X, n):
        """get indicies of random minibatch samples (each of size n) from X"""
        idx = np.random.permutation(range(len(X)))
        batches = range(0, len(X)+1, n)

        # remainder if minibatch size does not divide into total number of samples
        remainder = len(X) - batches[-1]
        if remainder > 0:
            batches.append(batches[-1] + remainder-1)

        # collect a list of arrays with indicies, one array for each minibatch
        batched_idx = []
        for i in range(len(batches)):
            if batches[-1] == batches[i]:
                break
            batched_idx.append(idx[batches[i]:batches[i+1]])

        if self.debug:
            batched_idx = batched_idx[:100]

        return(batched_idx)


    def renormalize(self, X):
        """scale correlations between [0 1]"""
        return((X + 1) / 2)


    def activate(self, layer):
        return(tf.nn.tanh(layer, name='X_encode'))


    def transform(self, data):
        """passes input data through trained autoencoder"""

        sess = tf.Session()

        X = tf.constant(data, dtype=tf.float32)
        for w, b in zip(self.weights, self.biases):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(X, weight) + bias
            X = self.activate(layer)

        return(X.eval(session=sess))


    def fit(self, data):
        # greedy layer-wise training
        for i in range(self.depth) :
            print('training layer={} with {} hidden nodes'.format(i+1, self.hid[i]))
            if i == 0:
                self.train_layer(self.hid[i], data)
            else:
                self.train_layer(self.hid[i], self.encoded_data.root.encoded_data)


    def train_layer(self, hid, data):
        """trains the next hidden layer of the autoencoder"""

        m, n = data.shape

        # autoencoder activation graph
        sess = tf.Session()
        X = tf.placeholder(dtype=tf.float32, shape=[None, n], name='X')
        X_corrupt = tf.placeholder(dtype=tf.float32, shape=[None, n], name='X_corrupt')

        #loss = tf.Variable(np.inf, dtype=tf.float32, name='loss')
        encode = {'weights': tf.Variable(tf.truncated_normal([n, hid], stddev=self.weightstd, dtype=tf.float32)),
                  'biases':  tf.Variable(tf.truncated_normal([hid], stddev=self.weightstd, dtype=tf.float32))}
        decode = {'biases':  tf.Variable(tf.truncated_normal([n], stddev=self.weightstd, dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])} # weight share

        X_encode = self.activate(tf.matmul(X_corrupt, encode['weights']) + encode['biases'])
        X_decode = tf.matmul(X_encode, decode['weights']) + decode['biases']

        loss = tf.losses.mean_squared_error(X, X_decode)
        train_op = tf.train.AdamOptimizer(self.lr, name='train_op').minimize(loss)

        msg1 = tf.Print(encode['weights'], [encode['weights']], "encode weights: ")
        msg2 = tf.Print(loss, [loss], "loss: ")
        #msg2 = tf.Print(encode['biases'], [encode['biases']], "encode biases: ")
        #msg3 = tf.Print(decode['biases'], [decode['biases']], "decode biases: ")
        #msg4 = tf.Print(X, [X], "X raw: ")
        #msg5 = tf.Print(X_encode, [X_encode], "X_encode: ")
        #msg6 = tf.Print(X_decode, [X_decode], "X_decode: ")

        # run the activation graph
        sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()

        # epochs
        for i in range(self.epoch):
            batch_idx = self.get_batches(data, self.batch_size)
            print('{}: epoch {}/{} started'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i+1, self.epoch))

            # batches
            n_batches = len(batch_idx)
            for j in range(n_batches):
                X_batch = data[batch_idx[j], :]
                X_batch = self.clean(X_batch)
                #X_batch = renormalize(X_batch)
                X_batch_corrupt = self.corrupt(X_batch)

                # backprop, compute loss
                fp, l = sess.run([train_op, loss], feed_dict={X: X_batch, X_corrupt: X_batch_corrupt})
                self.training_loss.append(l)

                if self.debug:
                    #sess.run([msg1, msg2, msg3, msg4, msg5, msg6], feed_dict={X: X_batch, X_corrupt: X_batch_corrupt})
                    sess.run([msg1, msg2], feed_dict={X: X_batch, X_corrupt: X_batch_corrupt})

                if j % 50 == 0:
                    #saver.save(sess, 'layer-{}-epoch-{}-batch'.format(self.current_depth, i), global_step=j)
                    print('{}: batch={}/{}, loss={}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), j+1, n_batches, l))

        # runs the trained autoencoder on the full input dataset in batches
        # saves encoded data in hdf5 format
        # saves learned weights & biases (once)
        # TODO X_corrupt is a misleading name here -- should rethink namespace

        for i in range(n_batches):
            X_batch = data[batch_idx[i], :]
            if i == 0:
                X_transformed, w_learned, b_learned = sess.run([X_encode, encode['weights'], encode['biases']], feed_dict={X_corrupt: X_batch})

                fid = tables.open_file("encoded_data_hid_{}.h5".format(self.current_depth), "w")
                filters = tables.Filters(complevel=5, complib='blosc')
                encoded_data = fid.create_earray(fid.root, 'encoded_data',
                    atom=tables.Atom.from_dtype(X_transformed.dtype),
                    shape=(0, X_transformed.shape[-1]),
                    expectedrows=(len(X_transformed)))
                encoded_data.append(X_transformed)

                self.weights.append(w_learned)
                self.biases.append(b_learned)
            else:
                X_transformed = sess.run(X_encode, feed_dict={X_corrupt: X_batch})
                encoded_data.append(X_transformed)

        fid.close()
        sess.close()

        # open up a pointer to the encoded data
        if self.current_depth > 1:
            self.encoded_data.root.encoded_data.close()
        self.encoded_data = tables.open_file("encoded_data_hid_{}.h5".format(self.current_depth), mode='r')
        X_transformed = self.encoded_data.root.encoded_data
        self.current_depth += 1

# report stats for full batch
#plt.plot(training_loss)
#plt.savefig('loss_epoch_{}.pdf'.format(i+1))
#plt.close()

