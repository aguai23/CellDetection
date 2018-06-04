import tensorflow as tf
from model.ops import conv2d, fc
from util.layers import pixel_wise_softmax_2
import numpy as np
import logging


def inception_module_origin(inputs, scope=""):
    """
    The original inception module, change 5x5 to two 3x3
    :param inputs: inputs with channel of 192
    :param scope: scope name
    :return: tensor
    """
    # 61 * 61 * 192
    with tf.variable_scope(scope):

        with tf.variable_scope('branch1x1'):
            branch1x1 = conv2d(inputs, 64, [1, 1])

        with tf.variable_scope('branch5x5'):
            branch5x5 = conv2d(inputs, 48, [1, 1])
            branch5x5 = conv2d(branch5x5, 64, [5, 5])

        with tf.variable_scope('branch3x3'):
            branch3x3 = conv2d(inputs, 64, [1, 1])
            branch3x3 = conv2d(branch3x3, 96, [3, 3])
            branch3x3 = conv2d(branch3x3, 96, [3, 3])

        with tf.variable_scope('branch_pool'):
            branch_pool = tf.nn.avg_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
            branch_pool = conv2d(branch_pool, 64, [1, 1])

        net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        return net


def inception_reduction_v1(inputs, scope=""):
    """
    module used for reduce dimension, as illustrated in figure 10 in the paper
    :param inputs: tensor of shape 288
    :param scope: scope name
    :return: tensor
    """
    with tf.variable_scope(scope):
        with tf.variable_scope("branch3x3"):
            branch3x3 = conv2d(inputs, 384, [3, 3], stride=2, padding='VALID')

        with tf.variable_scope("branch3x3db"):
            branch3x3db = conv2d(inputs, 64, [1, 1])
            branch3x3db = conv2d(branch3x3db, 96, [3, 3])
            branch3x3db = conv2d(branch3x3db, 96, [3, 3], stride=2, padding='VALID')

        with tf.variable_scope("branch_pool"):
            branch_pool = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')
        net = tf.concat(axis=3, values=[branch3x3, branch3x3db, branch_pool])
        return net


def inception_reduction_v2(inputs, scope=""):
    """
    module used for reduce dimension, with different input and output shape
    :param inputs: inputs tensor of 768
    :param scope: scope name
    :return: output tensor of 1280
    """
    with tf.variable_scope(scope):

        with tf.variable_scope('branch3x3'):
            branch3x3 = conv2d(inputs, 192, [1, 1])
            branch3x3 = conv2d(branch3x3, 320, [3, 3], stride=2, padding='VALID')

        with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = conv2d(inputs, 192, [1, 1])
            branch7x7x3 = conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = conv2d(branch7x7x3, 192, [3, 3], stride=2, padding='VALID')

        with tf.variable_scope("branch_pool"):
            branch_pool = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
        return net


def inception_module_factorization(inputs, scope=""):
    """
    inception module after factorization of n x n convolutions, as illustrated in figure 6 in paper
    :param inputs: input tensor with 784 channel
    :param scope: scope name
    :return: tensor
    """
    with tf.variable_scope(scope):

        with tf.variable_scope("branch1x1"):
            branch1x1 = conv2d(inputs, 192, [1, 1])

        with tf.variable_scope("branch7x7"):
            branch7x7 = conv2d(inputs, 160, [1, 1])
            branch7x7 = conv2d(branch7x7, 160, [1, 7])
            branch7x7 = conv2d(branch7x7, 192, [7, 1])

        with tf.variable_scope("branch7x7db"):
            branch7x7db = conv2d(inputs, 160, [1, 1])
            branch7x7db = conv2d(branch7x7db, 160, [7, 1])
            branch7x7db = conv2d(branch7x7db, 160, [1, 7])
            branch7x7db = conv2d(branch7x7db, 160, [7, 1])
            branch7x7db = conv2d(branch7x7db, 192, [1, 7])

        with tf.variable_scope("branch_pool"):
            branch_pool = tf.nn.avg_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
            branch_pool = conv2d(branch_pool, 192, [1, 1])

        net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7db, branch_pool])
        return net


def inception_module_expanded(inputs, scope=""):

    with tf.variable_scope(scope):

        with tf.variable_scope('branch1x1'):
            branch1x1 = conv2d(inputs, 320, [1, 1])

        with tf.variable_scope('branch3x3'):
            branch3x3 = conv2d(inputs, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[conv2d(branch3x3, 384, [1, 3]),
                                                  conv2d(branch3x3, 384, [3, 1])])
        with tf.variable_scope('branch3x3db'):
            branch3x3db = conv2d(inputs, 448, [1, 1])
            branch3x3db = conv2d(branch3x3db, 384, [3, 3])
            branch3x3db = tf.concat(axis=3, values=[conv2d(branch3x3db, 384, [1, 3]),
                                                    conv2d(branch3x3db, 384, [3, 1])])

        with tf.variable_scope("branch_pool"):
            branch_pool = tf.nn.avg_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
            branch_pool = conv2d(branch_pool, 192, [1, 1])

        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3db, branch_pool])
        return net


def inception_v3(inputs, dropout_keep_prob=0.8, num_classes=2,
                 is_training=True, restore_logits=True, scope=""):

    with tf.name_scope(scope, "inception_v3", [inputs]):

        # 512 * 512 * 3
        conv0 = conv2d(inputs, 32, [3, 3], stride=2, scope='conv0', padding='VALID')

        # 255 * 255 * 32
        conv1 = conv2d(conv0, 32, [3, 3], scope='conv1', padding='VALID')

        # 253 * 253 * 32
        conv2 = conv2d(conv1, 64, [3, 3], scope='conv2')

        # 253 * 253 * 64
        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=scope)

        # 126 * 126 * 64
        conv3 = conv2d(conv2, 80, [1, 1], scope='conv3', padding='VALID')

        # 126 * 126 * 80
        conv4 = conv2d(conv3, 192, [3, 3], scope='conv4', padding='VALID')

        # 124 * 124 * 192
        with tf.name_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=scope)

        net = pool2
        for i in range(3):
            net = inception_module_origin(net, scope="inception_origin" + str(i+1))

        # 61 * 61 * 288
        net = inception_reduction_v1(net, scope="reduction_v1")

        # 30 * 30 * 768
        for i in range(4):
            net = inception_module_factorization(net, scope="inception_factorization" + str(i))

        # Auxiliary Head logits, input 30 * 30 * 768
        with tf.variable_scope("aux_logits"):
            aux_logits = tf.identity(net)
            aux_logits = tf.nn.avg_pool(aux_logits, [1, 5, 5, 1], [1, 3, 3, 1], padding='VALID')
            # 9 * 9 * 768
            aux_logits = conv2d(aux_logits, 128, [1, 1], scope='proj')
            shape = aux_logits.get_shape()
            aux_logits = conv2d(aux_logits, 768, shape[1:3], stddev=0.01, padding='VALID')
            dims = aux_logits.get_shape()[1:]
            k = dims.num_elements()
            aux_logits = tf.reshape(aux_logits, [-1, k])
            aux_logits = fc(aux_logits, num_classes, activation=None, stddev=0.001)

        net = inception_reduction_v2(net, scope="reduction_v2")

        # 14 * 14 * 1280
        for i in range(2):
            net = inception_module_expanded(net, scope="inception_expanded" + str(i))

        with tf.variable_scope("logits"):
            shape = net.get_shape()
            net = tf.nn.avg_pool(net, [1, shape[1], shape[2], 1], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.dropout(net, dropout_keep_prob, name="dropout")
            dims = net.get_shape()[1:].num_elements()
            net = tf.reshape(net, [-1, dims])
            logits = fc(net, num_classes, activation=None, scope="logits")

        return logits, aux_logits


class InceptionV3(object):

    def __init__(self, n_class=2, cost="cross_entropy"):
        tf.reset_default_graph()
        self.n_class = n_class
        self.summaries = True

        self.x = tf.placeholder("float", shape=[None, 512, 512, 3])
        self.y = tf.placeholder("float", shape=[None, n_class])

        self.keep_prob = tf.placeholder(tf.float32)

        self.logits, self.aux_logits = inception_v3(self.x, self.keep_prob)
        self.cost = self._get_cost(self.logits, self.aux_logits, cost)
        self.predictor = pixel_wise_softmax_2(self.logits)

    def _get_cost(self, logits, aux_logits,  cost_name):

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_aux_logits = tf.reshape(aux_logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])

        if cost_name == "cross_entropy":
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                          labels=flat_labels)) \
                   + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                            labels=flat_aux_logits))
        else:
            raise ValueError("unknown cost function ")

        return cost

    def predict(self, model_path, x_test):
        """
        predict data from trained model
        :param model_path:
        :param x_test:
        :return: prediction of shape [n, class_num]
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.restore(sess, model_path)
            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run([self.predictor], feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1})
        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """
        print(model_path)
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)