import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

###################
# parameters
###################

# dataset
flags.DEFINE_bool('finite', True, 'sinusoid, sinusoid_finite')
flags.DEFINE_integer('train_total_num_tasks', 100, 'total number of tasks for training with finite dataset')
flags.DEFINE_integer('test_total_num_tasks', 100, 'total number of tasks for evaluation')

# model options
flags.DEFINE_integer('seed', 10, 'random seed')
flags.DEFINE_integer('num_particles', 10, 'number of particles per task')
flags.DEFINE_integer('num_tasks', 10, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('few_k_shot', 5, 'K for K-shot learning')
flags.DEFINE_integer('val_k_shot', 5, 'validation')

# log and train option
flags.DEFINE_integer('num_epochs', 10000, 'num_epochs')

logdir = 'log'
figdir = 'fig'

if not os.path.exists(figdir):
    os.makedirs(figdir)

# make conditions
conditions = []

if FLAGS.finite:
    conditions.append('SinusoidFinite'+str(FLAGS.train_total_num_tasks))
    conditions.append('Test'+str(FLAGS.test_total_num_tasks))
else:
    conditions.append('SinusoidInfiniteTest'+str(FLAGS.test_total_num_tasks))
conditions.append('Epoch'+str(FLAGS.num_epochs))
conditions.append('T'+str(FLAGS.num_tasks))
conditions.append('M'+str(FLAGS.num_particles))
conditions.append('SEED'+str(FLAGS.seed))
conditions.append('TrainK'+str(FLAGS.few_k_shot))

# get log folder names
logs = os.listdir(logdir)
logs.reverse()

# get bmaml folder name
bmaml_folder = False
for log in logs:
    if 'BMAML' in log:
        cond_cnt = 0
        for cond in conditions:
            if cond in log:
                cond_cnt += 1
        if cond_cnt == len(conditions):
            bmaml_folder = log
            break

# get emaml folder name
emaml_folder = False
for log in logs:
    if 'EMAML' in log:
        cond_cnt = 0
        for cond in conditions:
            if cond in log:
                cond_cnt += 1
        if cond_cnt == len(conditions):
            emaml_folder = log
            break

print(bmaml_folder)
print(emaml_folder)
if bmaml_folder is False or emaml_folder is False:
    raise ValueError('there are no log for the options')

bmaml_file = logdir + '/' + bmaml_folder + '/results.pkl'
emaml_file = logdir + '/' + emaml_folder + '/results.pkl'

bmaml_data = []
with open(bmaml_file, 'rb') as f:
    bmaml_data = pkl.load(f)

emaml_data = []
with open(emaml_file, 'rb') as f:
    emaml_data = pkl.load(f)

plt.plot(bmaml_data[0], bmaml_data[2], '-', label='bmaml')
plt.plot(emaml_data[0], emaml_data[1], '-', label='emaml')
plt.legend()
#plt.ylim([1, 100])
plt.savefig(figdir+'/'+'_'.join(conditions)+'.png')
plt.close()

