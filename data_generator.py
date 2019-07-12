from collections import OrderedDict
import numpy as np
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


class SinusoidGenerator(object):
    def __init__(self, split_data=True):
        # data size
        self.split_data = split_data
        self.num_tasks = FLAGS.num_tasks
        self.few_k_shot = FLAGS.few_k_shot
        self.val_k_shot = FLAGS.val_k_shot
        if self.split_data:
            self.all_k_shot = FLAGS.few_k_shot + FLAGS.val_k_shot
            self.total_samples = self.all_k_shot + self.val_k_shot
        else:
            self.all_k_shot = None
            self.total_samples = self.few_k_shot + self.val_k_shot
        self.dim_input = 1
        self.dim_output = 1

        # train set
        self.amp_range_train = [0.1, 5.0]
        self.phs_range_train = [0, np.pi*FLAGS.phase]
        self.frq_range_train = [0.5, 1.0*FLAGS.freq]
        self.inp_range_train = [-5.0, 5.0]

        # data size
        self.dim_input = 1
        self.dim_output = 1

        # set valid/test tasks
        self.valid_tasks = self.generate_finite_dataset(total_num_tasks=int(FLAGS.test_total_num_tasks),
                                                        is_training=False)
        self.test_tasks = self.generate_finite_dataset(total_num_tasks=int(FLAGS.test_total_num_tasks),
                                                       is_training=False)
        self.valid_batch_idx = 0
        self.test_batch_idx = 0

        # set train tasks
        if FLAGS.finite:
            self.train_tasks = self.generate_finite_dataset(total_num_tasks=int(FLAGS.train_total_num_tasks),
                                                            is_training=True)
            self.train_batch_idx = 0
            self.generate_batch = self.generate_finite_batch
        else:
            self.generate_batch = self.generate_infinite_batch

    # generate finite dataset
    def generate_finite_dataset(self, total_num_tasks, is_training=True):
        # get range
        amp_range = self.amp_range_train
        phs_range = self.phs_range_train
        frq_range = self.frq_range_train
        inp_range = self.inp_range_train

        # sample tasks
        amp_list = np.random.uniform(low=amp_range[0], high=amp_range[1], size=[total_num_tasks])
        phs_list = np.random.uniform(low=phs_range[0], high=phs_range[1], size=[total_num_tasks])
        frq_list = np.random.uniform(low=frq_range[0], high=frq_range[1], size=[total_num_tasks])
        x_list = np.random.uniform(low=inp_range[0], high=inp_range[1], size=[total_num_tasks, self.total_samples, 1])
        y_list = np.zeros(shape=[total_num_tasks, self.total_samples, 1])
        z_list = np.zeros(shape=[total_num_tasks, self.total_samples, 1])

        # for each task
        for t in range(total_num_tasks):
            # sample noise
            z_list[t] = np.random.normal(loc=0.0, scale=FLAGS.noise_factor * amp_list[t], size=[self.total_samples, 1])

            # compute output
            y_list[t] = amp_list[t] * np.sin(frq_list[t] * x_list[t] - phs_list[t])

        # task dataset
        tasks = OrderedDict()
        tasks['x'] = x_list
        tasks['y'] = y_list
        tasks['z'] = z_list
        tasks['amp'] = amp_list
        tasks['phs'] = phs_list
        tasks['frq'] = frq_list
        tasks['size'] = total_num_tasks
        return tasks

    # get batch from finite dataset
    def generate_finite_batch(self,
                              is_training=True,
                              batch_idx=None,
                              inc_follow=True):
        # get dataset
        if is_training:
            task_list = self.train_tasks
            num_tasks = task_list['size']
        elif FLAGS.train:
            task_list = self.valid_tasks
            num_tasks = task_list['size']
        else:
            task_list = self.test_tasks
            num_tasks = task_list['size']

        # get batch-wise data
        if batch_idx is not None:
            x_list = task_list['x'][batch_idx:(batch_idx + self.num_tasks)]
            y_list = task_list['y'][batch_idx:(batch_idx + self.num_tasks)]
            z_list = task_list['z'][batch_idx:(batch_idx + self.num_tasks)]
        else:
            idx_list = np.arange(num_tasks)
            np.random.shuffle(idx_list)
            idx_list = idx_list[:self.num_tasks]
            x_list = task_list['x'][idx_list]
            y_list = task_list['y'][idx_list]
            z_list = task_list['z'][idx_list]

        # split data
        if self.split_data:
            follow_x = x_list[:, :self.few_k_shot]
            follow_y = y_list[:, :self.few_k_shot]
            follow_z = z_list[:, :self.few_k_shot]

            if inc_follow:
                leader_x = x_list[:, :self.all_k_shot]
                leader_y = y_list[:, :self.all_k_shot]
                leader_z = z_list[:, :self.all_k_shot]
            else:
                leader_x = x_list[:, self.few_k_shot:self.all_k_shot]
                leader_y = y_list[:, self.few_k_shot:self.all_k_shot]
                leader_z = z_list[:, self.few_k_shot:self.all_k_shot]

            valid_x = x_list[:, self.all_k_shot:]
            valid_y = y_list[:, self.all_k_shot:]
            valid_z = z_list[:, self.all_k_shot:]

            # add noise
            return [follow_x,
                    leader_x,
                    valid_x,
                    follow_y + follow_z,
                    leader_y + leader_z,
                    valid_y + valid_z if is_training else valid_y]
        else:
            train_x, valid_x = x_list[:, :self.few_k_shot], x_list[:, self.few_k_shot:]
            train_y, valid_y = y_list[:, :self.few_k_shot], y_list[:, self.few_k_shot:]
            train_z, valid_z = z_list[:, :self.few_k_shot], z_list[:, self.few_k_shot:]

            # add noise
            return [train_x,
                    valid_x,
                    train_y + train_z,
                    valid_y + valid_z if is_training else valid_y]

    # get batch from infinite dataset
    def generate_infinite_batch(self,
                                is_training=True,
                                batch_idx=None,
                                inc_follow=True):
        if is_training:
            # get range
            amp_range = self.amp_range_train
            phs_range = self.phs_range_train
            frq_range = self.frq_range_train
            inp_range = self.inp_range_train

            # sample tasks
            amp_list = np.random.uniform(low=amp_range[0], high=amp_range[1], size=[self.num_tasks])
            phs_list = np.random.uniform(low=phs_range[0], high=phs_range[1], size=[self.num_tasks])
            frq_list = np.random.uniform(low=frq_range[0], high=frq_range[1], size=[self.num_tasks])
            x_list = np.random.uniform(low=inp_range[0], high=inp_range[1], size=[self.num_tasks, self.total_samples, 1])
            y_list = np.zeros(shape=[self.num_tasks, self.total_samples, 1])
            z_list = np.zeros(shape=[self.num_tasks, self.total_samples, 1])

            # for each task
            for t in range(self.num_tasks):
                # sample noise
                z_list[t] = np.random.normal(loc=0.0, scale=FLAGS.noise_factor * amp_list[t], size=[self.total_samples, 1])

                # compute output
                y_list[t] = amp_list[t] * np.sin(frq_list[t] * x_list[t] - phs_list[t])

        else:
            if FLAGS.train:
                task_list = self.valid_tasks
                num_tasks = task_list['size']
            else:
                task_list = self.test_tasks
                num_tasks = task_list['size']

            # get batch-wise data
            if batch_idx is not None:
                x_list = task_list['x'][batch_idx:(batch_idx + self.num_tasks)]
                y_list = task_list['y'][batch_idx:(batch_idx + self.num_tasks)]
                z_list = task_list['z'][batch_idx:(batch_idx + self.num_tasks)]
            else:
                idx_list = np.arange(num_tasks)
                np.random.shuffle(idx_list)
                idx_list = idx_list[:self.num_tasks]
                x_list = task_list['x'][idx_list]
                y_list = task_list['y'][idx_list]
                z_list = task_list['z'][idx_list]

        # split data
        if self.split_data:
            follow_x = x_list[:, :self.few_k_shot]
            follow_y = y_list[:, :self.few_k_shot]
            follow_z = z_list[:, :self.few_k_shot]

            if inc_follow:
                leader_x = x_list[:, :self.all_k_shot]
                leader_y = y_list[:, :self.all_k_shot]
                leader_z = z_list[:, :self.all_k_shot]
            else:
                leader_x = x_list[:, self.few_k_shot:self.all_k_shot]
                leader_y = y_list[:, self.few_k_shot:self.all_k_shot]
                leader_z = z_list[:, self.few_k_shot:self.all_k_shot]

            valid_x = x_list[:, self.all_k_shot:]
            valid_y = y_list[:, self.all_k_shot:]
            valid_z = z_list[:, self.all_k_shot:]

            # add noise
            return [follow_x,
                    leader_x,
                    valid_x,
                    follow_y + follow_z ,
                    leader_y + leader_z,
                    valid_y + valid_z if is_training else valid_y]
        else:
            train_x, valid_x = x_list[:, :self.few_k_shot], x_list[:, self.few_k_shot:]
            train_y, valid_y = y_list[:, :self.few_k_shot], y_list[:, self.few_k_shot:]
            train_z, valid_z = z_list[:, :self.few_k_shot], z_list[:, self.few_k_shot:]

            # add noise
            return [train_x,
                    valid_x,
                    train_y + train_z,
                    valid_y + valid_z if is_training else valid_y]
