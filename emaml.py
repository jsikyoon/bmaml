from bnn import BNN
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


class EMAML:
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden=32,
                 num_layers=4,
                 num_particles=2,
                 max_test_step=5):
        # model size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.num_particles = num_particles

        # learning rate
        self.in_lr = tf.placeholder_with_default(input=FLAGS.in_lr,
                                                 name='in_lr',
                                                 shape=[])
        self.out_lr = tf.placeholder_with_default(input=FLAGS.out_lr,
                                                  name='out_lr',
                                                  shape=[])

        # for test time
        self.max_test_step = max_test_step

        # build model
        self.bnn = BNN(dim_input=self.dim_input,
                       dim_output=self.dim_output,
                       dim_hidden=self.dim_hidden,
                       num_layers=self.num_layers,
                       is_bnn=False)

        # init model
        self.construct_network_weights = self.bnn.construct_network_weights

        # forwarding
        self.forward_network = self.bnn.forward_network

        # init input data
        self.train_x = tf.placeholder(dtype=tf.float32, name='train_x')
        self.train_y = tf.placeholder(dtype=tf.float32, name='train_y')
        self.valid_x = tf.placeholder(dtype=tf.float32, name='valid_x')
        self.valid_y = tf.placeholder(dtype=tf.float32, name='valid_y')

        # init parameters
        self.W_network_particles = None

    # build model
    def construct_model(self,
                        is_training=True):
        print('start model construction')
        # init model
        with tf.variable_scope('model', reuse=None) as training_scope:
            # init parameters
            if is_training or self.W_network_particles is None:
                # network parameters
                self.W_network_particles = [self.construct_network_weights(scope='network{}'.format(p_idx))
                                            for p_idx in range(self.num_particles)]
            else:
                training_scope.reuse_variables()

            # set number of follower steps
            if is_training:
                max_update_step = FLAGS.in_step
            else:
                max_update_step = max(FLAGS.in_step, self.max_test_step)

            # task-wise inner loop
            def fast_learn_one_task(inputs):
                # decompose input data
                [train_x, valid_x,
                 train_y, valid_y] = inputs

                ##########
                # update #
                ##########
                # init meta loss
                meta_loss = []

                # get the follow particles
                WW_update = [OrderedDict(zip(W_dic.keys(), W_dic.values()))
                             for W_dic in self.W_network_particles]

                # for each step
                step_train_loss = [None] * (max_update_step + 1)
                step_valid_loss = [None] * (max_update_step + 1)
                step_train_pred = [None] * (max_update_step + 1)
                step_valid_pred = [None] * (max_update_step + 1)
                for s_idx in range(max_update_step + 1):
                    # for each particle
                    train_z_list = []
                    valid_z_list = []
                    train_mse_list = []
                    valid_mse_list = []
                    for p_idx in range(FLAGS.num_particles):
                        # compute prediction
                        train_z_list.append(self.forward_network(x=train_x, W_dict=WW_update[p_idx]))
                        valid_z_list.append(self.forward_network(x=valid_x, W_dict=WW_update[p_idx]))

                        # compute mse data
                        train_mse_list.append(self.bnn.mse_data(predict_y=train_z_list[-1], target_y=train_y))
                        valid_mse_list.append(self.bnn.mse_data(predict_y=valid_z_list[-1], target_y=valid_y))

                        # update
                        if s_idx < max_update_step:
                            # compute loss and gradient
                            particle_loss = tf.reduce_mean(train_mse_list[-1])
                            dWp = tf.gradients(ys=particle_loss,
                                               xs=list(WW_update[p_idx].values()))

                            # stop gradient to avoid second order
                            if FLAGS.stop_grad:
                                dWp = [tf.stop_gradient(grad) for grad in dWp]

                            # re-order
                            dWp = OrderedDict(zip(WW_update[p_idx].keys(), dWp))

                            # for each param
                            param_names = []
                            param_vals = []
                            for key in list(WW_update[p_idx].keys()):
                                if FLAGS.in_grad_clip > 0:
                                    grad = tf.clip_by_value(dWp[key], -FLAGS.in_grad_clip, FLAGS.in_grad_clip)
                                else:
                                    grad = dWp[key]
                                param_names.append(key)
                                param_vals.append(WW_update[p_idx][key] - self.in_lr * grad)
                            WW_update[p_idx] = OrderedDict(zip(param_names, param_vals))
                        else:
                            # meta-loss
                            meta_loss.append(tf.reduce_mean(valid_mse_list[-1]))

                    # aggregate particle results
                    step_train_loss[s_idx] = tf.reduce_mean([tf.reduce_mean(train_mse) for train_mse in train_mse_list])
                    step_valid_loss[s_idx] = tf.reduce_mean([tf.reduce_mean(valid_mse) for valid_mse in valid_mse_list])
                    step_train_pred[s_idx] = tf.concat([tf.expand_dims(train_z, 0) for train_z in train_z_list], axis=0)
                    step_valid_pred[s_idx] = tf.concat([tf.expand_dims(valid_z, 0) for valid_z in valid_z_list], axis=0)

                # sum meta-loss over particles
                meta_loss = tf.reduce_sum(meta_loss)
                return [step_train_loss,
                        step_valid_loss,
                        step_train_pred,
                        step_valid_pred,
                        meta_loss]

            # set output type
            out_dtype = [[tf.float32] * (max_update_step + 1),
                         [tf.float32] * (max_update_step + 1),
                         [tf.float32] * (max_update_step + 1),
                         [tf.float32] * (max_update_step + 1),
                         tf.float32]

            # compute over tasks
            result = tf.map_fn(fast_learn_one_task,
                               elems=[self.train_x, self.valid_x,
                                      self.train_y, self.valid_y],
                               dtype=out_dtype,
                               parallel_iterations=FLAGS.num_tasks)

            # unroll result
            full_step_train_loss = result[0]
            full_step_valid_loss = result[1]
            full_step_train_pred = result[2]
            full_step_valid_pred = result[3]
            full_meta_loss = result[4]

            # for training
            if is_training:
                # summarize results
                self.total_train_loss = [tf.reduce_mean(full_step_train_loss[j])
                                         for j in range(FLAGS.in_step + 1)]
                self.total_valid_loss = [tf.reduce_mean(full_step_valid_loss[j])
                                         for j in range(FLAGS.in_step + 1)]
                self.total_meta_loss = tf.reduce_mean(full_meta_loss)

                # prediction
                self.total_train_z_list = full_step_train_pred
                self.total_valid_z_list = full_step_valid_pred

                ###############
                # meta update #
                ###############
                update_params_list = []
                update_params_name = []

                # get params
                for p in range(FLAGS.num_particles):
                    for name in self.W_network_particles[0].keys():
                        update_params_name.append([p, name])
                        update_params_list.append(self.W_network_particles[p][name])

                # set optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=self.out_lr)

                # compute gradient
                gv_list = optimizer.compute_gradients(loss=self.total_meta_loss,
                                                      var_list=update_params_list)

                # gradient clipping
                if FLAGS.out_grad_clip > 0:
                    gv_list = [(tf.clip_by_value(grad, -FLAGS.out_grad_clip, FLAGS.out_grad_clip), var)
                               for grad, var in gv_list]

                # optimizer
                self.metatrain_op = optimizer.apply_gradients(gv_list)
            else:
                # summarize results
                self.eval_train_loss = [tf.reduce_mean(full_step_train_loss[j])
                                        for j in range(max_update_step + 1)]
                self.eval_valid_loss = [tf.reduce_mean(full_step_valid_loss[j])
                                        for j in range(max_update_step + 1)]

                # prediction
                self.eval_train_z_list = full_step_train_pred
                self.eval_valid_z_list = full_step_valid_pred
        print('end of model construction')


