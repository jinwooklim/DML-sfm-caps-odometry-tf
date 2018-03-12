"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf

from config import cfg
from utils_caps import get_batch_data
from utils_caps import softmax
from utils_caps import reduce_sum
from capsLayer import CapsLayer


epsilon = 1e-9


class CapsNet(object):
    def __init__(self):
        pass

    def build_all(self, conv1):
        if cfg.is_training:
            self.X, self.labels = get_batch_data(cfg.dataset_dir, cfg.capsdata_dir, cfg.batch_size) # (4,4), (4,)
            self.Y = tf.one_hot(self.labels, depth=cfg.num_of_class, axis=1, dtype=tf.float32)
            self.build_arch(conv1)
            self.loss()
            self._summary()
        else:
            #self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 4))
            #self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
            #self.Y = tf.one_hot(self.labels, depth=cfg.num_of_class, axis=1, dtype=tf.float32)
            self.build_arch(conv1)
    
    
    def build_arch(self, conv1):
        with tf.variable_scope('Conv1_layer'):
            self.conv1 = conv1
        
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            self.caps1 = primaryCaps(self.conv1, kernel_size=9, stride=2)

        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=cfg.num_of_class, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(self.caps1)

        if cfg.is_training:
            # Decoder structure in Fig. 2
            # 1. Do masking, how:
            with tf.variable_scope('Masking'):
                #print("----- Masking step -----")
                # a). calc ||v_c||, then do softmax(||v_c||)
                # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
                self.softmax_v = softmax(self.v_length, axis=1)
                assert self.softmax_v.get_shape() == [cfg.batch_size, cfg.num_of_class, 1, 1]

                # b). pick out the index of max softmax val of the 10 caps
                # [batch_size, 10, 1, 1] => [batch_size] (index)
                self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
                assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
                self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

                # Method 1.
                if not cfg.mask_with_y:
                    # c). indexing
                    # It's not easy to understand the indexing process with argmax_idx
                    # as we are 3-dim animal
                    masked_v = []
                    for batch_size in range(cfg.batch_size):
                        v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                        masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                    self.masked_v = tf.concat(masked_v, axis=0)
                    assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
                # Method 2. masking with true label, default mode
                else:
                    self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, cfg.num_of_class, 1)))
                    self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

            # 2. Reconstructe the MNIST images with 3 FC layers
            # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
            with tf.variable_scope('Decoder'):
                vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
                fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
                assert fc1.get_shape() == [cfg.batch_size, 512]
                fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
                assert fc2.get_shape() == [cfg.batch_size, 1024]
                fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=512)
                fc4 = tf.contrib.layers.fully_connected(fc3, num_outputs=128)
                self.decoded = tf.contrib.layers.fully_connected(fc4, num_outputs=4, activation_fn=tf.sigmoid)
        else:
            # Only Decoding part
            v_length, prediction = predict(self.caps2)
            # Masking
            with tf.variable_scope('Masking'):
                # batch size of predictions (labels)
                candid = []
                for index in range(cfg.batch_size):
                    v = self.caps2[index][prediction[index], :] # [16, 1]
                    candid.append(tf.reshape(v, shape=(1, 1, 16, 1)))
                candid = tf.concat(candid, axis=0)
                assert candid.get_shape() == [cfg.batch_size, 1, 16, 1]
            
            # Reconstruct batch size of images
            with tf.variable_scope('Decoder'):
                v = tf.reshape(candid, shape=(cfg.batch_size, -1))
                fc1 = tf.contrib.layers.fully_connected(v, num_outputs=512)
                assert fc1.get_shape() == [cfg.batch_size, 512]
                fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
                assert fc2.get_shape() == [cfg.batch_size, 1024]
                fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=512)
                fc4 = tf.contrib.layers.fully_connected(fc3, num_outputs=128)
                self.decoded = tf.contrib.layers.fully_connected(fc4, num_outputs=4, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        #assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]
        assert max_l.get_shape() == [cfg.batch_size, cfg.num_of_class, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        #print(T_c.get_shape())
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        #recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        #train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


def predict(caps2):
    '''
    Return prediction with argmax
    '''
    with tf.variable_scope('label_prediction'):
        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + epsilon)
        assert v_length.get_shape() == [cfg.batch_size, cfg.num_of_class, 1, 1]
        softmax_v = tf.nn.softmax(v_length, dim=1)
        # index of max softmax val among the 10 digit
        prediction = tf.to_int32(tf.argmax(softmax_v, axis=1))
        assert prediction.get_shape() == [cfg.batch_size, 1, 1]
        prediction = tf.reshape(prediction, shape=(cfg.batch_size, ))
        return v_length, prediction
