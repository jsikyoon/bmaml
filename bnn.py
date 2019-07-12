import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


# network
class BNN(object):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden,
                 num_layers,
                 is_bnn=True):
        # set model size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        # for bayesian
        self.is_bnn = is_bnn

    def construct_network_weights(self, scope='network'):
        # init
        params = OrderedDict()

        # initializer
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        # layer 1
        params['w1'] = tf.get_variable(name=scope + '_w1',
                                       shape=[self.dim_input, self.dim_hidden],
                                       initializer=fc_initializer)
        params['b1'] = tf.Variable(name=scope + '_b1',
                                   initial_value=tf.random_normal([self.dim_hidden], 0.0, 0.01))

        # layer l
        for l in range(self.num_layers):
            if l < (self.num_layers - 1):
                dim_output = self.dim_hidden
            else:
                dim_output = self.dim_output
            params['w{}'.format(l + 2)] = tf.get_variable(name=scope + '_w{}'.format(l + 2),
                                                          shape=[self.dim_hidden, dim_output],
                                                          initializer=fc_initializer)            
            params['b{}'.format(l + 2)] = tf.Variable(name=scope + '_b{}'.format(l + 2),
                                                      initial_value=tf.random_normal([dim_output], 0.0, 0.01))

        if self.is_bnn:
            init_val = np.random.normal(-np.log(FLAGS.m_l),  0.001, [1])
            params['log_lambda'] = tf.Variable(name=scope + "_log_lambda",
                                               initial_value=init_val,
                                               dtype=tf.float32)

            print('log_lambda: ', init_val)

            init_val = np.random.normal(-np.log(FLAGS.m_g),  0.001, [1])
            params['log_gamma'] = tf.Variable(name=scope + "_log_gamma",
                                              initial_value=init_val,
                                              dtype=tf.float32)

            print('log_gamma: ', init_val)

        return params

    # data log-likelihood
    def log_likelihood_data(self, predict_y, target_y, log_gamma):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        # error
        error_y = predict_y - target_y

        # compute log-prob
        log_lik_data = 0.5 * log_gamma - 0.5 * tf.exp(log_gamma) * tf.square(error_y)
        return log_lik_data

    # weight log-prior
    def log_prior_weight(self, W_dict):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        # convert into vector list
        W_vec = self.dicval2vec(W_dict)

        # get lambda, gamma
        log_lambda = tf.reshape(W_vec[-2], (1,))
        log_gamma = tf.reshape(W_vec[-1], (1,))

        # get only weights
        W_vec = W_vec[:-2]
        num_params = tf.cast(W_vec.shape[0], tf.float32)

        # get data log-prior
        log_prior_gamma = (FLAGS.a_g - 1) * log_gamma - FLAGS.b_g * tf.exp(log_gamma) + log_gamma

        # get weight log-prior
        W_diff = W_vec
        log_prior_w = 0.5 * num_params * log_lambda - 0.5 * tf.exp(log_lambda) * tf.reduce_sum(W_diff ** 2)
        log_prior_lambda = (FLAGS.a_l - 1) * log_lambda - FLAGS.b_l * tf.exp(log_lambda) + log_lambda

        return log_prior_w, log_prior_gamma, log_prior_lambda

    # mse data
    def mse_data(self, predict_y, target_y):
        return tf.reduce_sum(tf.square(predict_y - target_y), axis=1)

    # forward
    def forward_network(self, x, W_dict):
        hid = tf.nn.relu(tf.matmul(x, W_dict['w1']) + W_dict['b1'])
        for l in range(self.num_layers):
            hid = tf.matmul(hid, W_dict['w{}'.format(l + 2)]) + W_dict['b{}'.format(l + 2)]
            if l < (self.num_layers - 1):
                hid = tf.nn.relu(hid)
        return hid

    # list of params into vec
    def list2vec(self, list_in):
        return tf.concat([tf.reshape(ww, [-1]) for ww in list_in], axis=0)

    # vec param into dic
    def vec2dic(self, W_vec):
        if self.is_bnn:
            log_lambda = tf.reshape(W_vec[-2], (1,))
            log_gamma = tf.reshape(W_vec[-1], (1,))
            W_vec = W_vec[:-2]

            W_dic = self.network_weight_vec2dict(W_vec)

            W_dic['log_lambda'] = log_lambda
            W_dic['log_gamma'] = log_gamma
        else:
            W_dic = self.network_weight_vec2dict(W_vec)
        return W_dic

    # vec param into dic
    def network_weight_vec2dict(self, W_vec):
        W_dic = OrderedDict()
        dim_list = [self.dim_input] + [self.dim_hidden] * self.num_layers + [self.dim_output]

        # for each layer
        for l in range(len(dim_list) - 1):
            # input / output dim
            dim_input, dim_output = dim_list[l], dim_list[l + 1]

            # reshape into param
            W_dic['w{}'.format(l + 1)] = tf.reshape(W_vec[:(dim_input * dim_output)], [dim_input, dim_output])
            W_dic['b{}'.format(l + 1)] = W_vec[(dim_input * dim_output):((dim_input * dim_output) + dim_output)]

            # remove loaded param
            if l < (len(dim_list) - 2):
                W_vec = W_vec[((dim_input * dim_output) + dim_output):]
        return W_dic

    # dic to vec
    def dicval2vec(self, dic):
        return tf.concat([tf.reshape(val, [-1]) for val in dic.values()], axis=0)
