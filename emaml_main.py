"""
Examples of running script

# 5 shot sinusoid regression (|T|=100) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=10000

# 10 shot sinusoid regression (|T|=100) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=10000

# 5 shot sinusoid regression (|T|=1000) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=1000

# 10 shot sinusoid regression (|T|=1000) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=1000
"""

import time
import os
import random
import numpy as np
import pickle as pkl
import tensorflow as tf
from datetime import datetime
from collections import OrderedDict
from tensorflow.python.platform import flags
import utils
from emaml import EMAML
from data_generator import SinusoidGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FLAGS = flags.FLAGS

# dataset
flags.DEFINE_bool('finite', True, 'sinusoid, sinusoid_finite')
flags.DEFINE_integer('train_total_num_tasks', 100, 'total number of tasks for training with finite dataset')
flags.DEFINE_integer('test_total_num_tasks', 100, 'total number of tasks for evaluation')
flags.DEFINE_float('noise_factor', 0.01, 'noise_factor')
flags.DEFINE_float('phase', 2.0, 'phase')
flags.DEFINE_float('freq', 2.0, 'freq')

# model options
flags.DEFINE_integer('seed', 10, 'random seed')
flags.DEFINE_integer('num_particles', 10, 'number of particles per task')
flags.DEFINE_integer('num_tasks', 10, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('few_k_shot', 5, 'for follower (K for K-shot learning)')
flags.DEFINE_integer('val_k_shot', 5, 'just for evaluation')
flags.DEFINE_integer('in_step', 1, 'number of updates for inner')
flags.DEFINE_float('in_grad_clip', 0.0, 'gradients clip')
flags.DEFINE_float('out_grad_clip', 0.0, 'gradients clip')
flags.DEFINE_float('in_lr', 1e-3, 'step size alpha for inner gradient update.')
flags.DEFINE_float('out_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_float('decay_lr', 0.98, 'learning rate decay')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization')
flags.DEFINE_integer('dim_hidden', 40, 'num filters')
flags.DEFINE_integer('num_layers', 3, 'num layers')

# log and train option
flags.DEFINE_integer('num_epochs', 10000, 'num_epochs')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_string('gpu', '-1', 'id of the gpu to use in the local machine')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
os.environ["TZ"] = 'EST'  # The default time zone of EAI servers is UST. So, change it to EST
time.tzset()

print(FLAGS.flag_values_dict())

# interval
PRINT_INTERVAL = 10
TEST_PRINT_INTERVAL = 100


def train(model, dataset, saver, sess, config_str):
    # set log dir
    experiment_dir = FLAGS.logdir + '/' + config_str

    # set summary writer
    train_writer = tf.summary.FileWriter(experiment_dir, sess.graph)
    print('Done initializing, starting training.')

    # total number of iteration for each epoch
    num_iters_per_epoch = int(FLAGS.train_total_num_tasks / FLAGS.num_tasks)

    if not FLAGS.finite:
        num_iters_per_epoch = 1

    # init train results
    inner_train_loss = []
    inner_valid_loss = []
    meta_loss = []

    # init test results
    test_itr_list = []
    test_valid_loss_list = []
    best_test_loss = 1000.0
    best_test_iter = 0

    # for each epoch
    itr = 0
    for e_idx in range(FLAGS.num_epochs):
        # for each batch tasks
        for b_idx in range(num_iters_per_epoch):
            # count iter
            itr += 1

            # load data
            [train_x, valid_x,
             train_y, valid_y] = dataset.generate_batch(is_training=True,
                                                        batch_idx=None)

            # set input
            out_lr = FLAGS.out_lr * FLAGS.decay_lr ** (float(itr - 1) / float(FLAGS.num_epochs * num_iters_per_epoch / 100))
            feed_in = OrderedDict()
            feed_in[model.out_lr] = out_lr
            feed_in[model.train_x] = train_x
            feed_in[model.valid_x] = valid_x
            feed_in[model.train_y] = train_y
            feed_in[model.valid_y] = valid_y

            # set output op
            fetch_out = [model.metatrain_op,
                         model.total_train_loss,
                         model.total_valid_loss,
                         model.total_meta_loss]

            # run
            result = sess.run(fetch_out, feed_in)[1:]

            # aggregate results
            inner_train_loss.append(result[0])
            inner_valid_loss.append(result[1])
            meta_loss.append(result[2])

            # print
            if itr % PRINT_INTERVAL == 0:
                inner_train_loss = np.stack(inner_train_loss).mean(axis=0)
                inner_valid_loss = np.stack(inner_valid_loss).mean(axis=0)
                meta_loss = np.stack(meta_loss).mean(axis=0)

                print('======================================')
                print('exp: ', config_str)
                print('epoch: ', e_idx, ' total iter: ', itr)
                print('--------------------------------------')
                print('train_loss: ', inner_train_loss)
                print('valid_loss: ', inner_valid_loss)
                print('--------------------------------------')
                print('meta_loss: ', meta_loss)
                print('out_lr: ', out_lr)
                print('--------------------------------------')
                print('best_test_loss: ', best_test_loss, '({})'.format(best_test_iter))

                # reset
                inner_train_loss = []
                inner_valid_loss = []
                meta_loss = []

            # compute meta-validation error
            if itr % TEST_PRINT_INTERVAL == 0:
                eval_train_loss_list = []
                eval_valid_loss_list = []

                # set output
                fetch_out = [model.eval_train_loss[:(FLAGS.in_step + 1)],
                             model.eval_valid_loss[:(FLAGS.in_step + 1)]]

                # for each batch
                for i in range(int(FLAGS.test_total_num_tasks/FLAGS.num_tasks)):
                    # load data
                    [train_x, valid_x,
                     train_y, valid_y] = dataset.generate_batch(is_training=False,
                                                                batch_idx=i * FLAGS.num_tasks)

                    # set input
                    feed_in = OrderedDict()
                    feed_in[model.train_x] = train_x
                    feed_in[model.valid_x] = valid_x
                    feed_in[model.train_y] = train_y
                    feed_in[model.valid_y] = valid_y

                    # compute results
                    result = sess.run(fetch_out, feed_in)
                    eval_train_loss_list.append(result[0])
                    eval_valid_loss_list.append(result[1])

                # aggregate results
                eval_train_loss = np.stack(eval_train_loss_list).mean(axis=0)
                eval_valid_loss = np.stack(eval_valid_loss_list).mean(axis=0)

                # print out
                print('======================================')
                print('Eval')
                print('--------------------------------------')
                print('exp: ', config_str)
                print('epoch: ', e_idx, ' total iter: ', itr)
                print('--------------------------------------')
                print('train_loss: ', eval_train_loss)
                print('valid_loss: ', eval_valid_loss)

                # save results
                test_itr_list.append(itr)
                test_valid_loss_list.append(eval_valid_loss[-1])

                pkl.dump([test_itr_list, test_valid_loss_list],
                         open(experiment_dir + '/' + 'results.pkl', 'wb'))
                plt.title('valid loss during training')
                plt.plot(test_itr_list, test_valid_loss_list, '-', label='test loss')
                plt.savefig(experiment_dir + '/' + 'test_loss.png')
                plt.close()

                if best_test_loss > test_valid_loss_list[-1]:
                    best_test_loss = test_valid_loss_list[-1]
                    best_test_iter = itr
                    saver.save(sess, experiment_dir + '/' + 'best_model')


def test(model, dataset, sess, inner_lr):
    # for each batch
    eval_valid_loss_list = []
    for i in range(int(FLAGS.test_total_num_tasks/FLAGS.num_tasks)):
        # load data
        [train_x, valid_x,
         train_y, valid_y] = dataset.generate_batch(is_training=False,
                                                    batch_idx=i * FLAGS.num_tasks)

        # set input
        feed_in = OrderedDict()
        feed_in[model.in_lr] = inner_lr
        feed_in[model.train_x] = train_x
        feed_in[model.valid_x] = valid_x
        feed_in[model.train_y] = train_y
        feed_in[model.valid_y] = valid_y

        # result
        eval_valid_loss_list.append(sess.run(model.eval_valid_loss, feed_in))

    # aggregate results
    eval_valid_loss_list = np.array(eval_valid_loss_list)
    eval_valid_loss_mean = np.mean(eval_valid_loss_list, axis=0)
    return eval_valid_loss_mean


def main():
    # set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # set exp name
    fname_args = []
    if FLAGS.finite:
        fname_args += [('train_total_num_tasks', 'SinusoidFinite')]
        fname_args += [('test_total_num_tasks', 'Test')]
    else:
        fname_args += [('test_total_num_tasks', 'SinusoidInfiniteTest')]

    fname_args += [('num_epochs', 'Epoch'),
                   ('num_tasks', 'T'),
                   ('seed', 'SEED'),
                   ('noise_factor', 'Noise'),
                   ('num_particles', 'M'),
                   ('dim_hidden', 'H'),
                   ('num_layers', 'L'),
                   ('phase', 'PHS'),
                   ('freq', 'FRQ'),
                   ('few_k_shot', 'TrainK'),
                   ('val_k_shot', 'ValidK'),
                   ('in_step', 'InStep'),
                   ('in_grad_clip', 'InGrad'),
                   ('out_grad_clip', 'OutGrad'),
                   ('in_lr', 'InLr'),
                   ('out_lr', 'OutLr'),
                   ('decay_lr', 'DecLr')]

    config_str = utils.experiment_string2(FLAGS.flag_values_dict(), fname_args, separator='_')
    config_str = str(time.mktime(datetime.now().timetuple()))[:-2] + '_EMAML' + config_str
    print(config_str)

    # get data generator
    dataset = SinusoidGenerator(split_data=False)

    # get dataset size
    dim_output = dataset.dim_output
    dim_input = dataset.dim_input

    # init model
    model = EMAML(dim_input=dim_input,
                  dim_output=dim_output,
                  dim_hidden=FLAGS.dim_hidden,
                  num_layers=FLAGS.num_layers,
                  num_particles=FLAGS.num_particles,
                  max_test_step=10)

    # init model
    model.construct_model(is_training=True)

    # for testing
    model.construct_model(is_training=False)

    # set summ ops
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=1)

    # open session
    sess = tf.InteractiveSession()

    # init model
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.train:
        # start training
        train(model, dataset, saver, sess, config_str)

if __name__ == "__main__":
    main()

