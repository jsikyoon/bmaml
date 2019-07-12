# Baysian Model-Agnostic Meta-Learning

This repository contains implementations of the paper, [Bayesian Model-Agnostic Meta-Learning (Jaesik Yoon and Taesup Kim et al., NuerIPS 2018)](https://arxiv.org/abs/1806.03836). It includes code for running the sinusoid regression task described in the paper.

To comparison with MAML and Ensemble MAML, we implemented emaml_main.py and emaml.py. 
With the setting, num_particle=1, you can run MAML on emaml_main.py.

For the reinforcement learning experiments, plese see [this repository](https://github.com/jaesik817/bmaml_rl).

## Quick run
The instructions for the experiments described in the paper are at the top of each main script as followed.

* Examples of BMAML running script
```
# 5 shot sinusoid regression (|T|=100) with 10 particles
python bmaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=10000

# 10 shot sinusoid regression (|T|=100) with 10 particles
python bmaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=10000

# 5 shot sinusoid regression (|T|=1000) with 10 particles
python bmaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=1000

# 10 shot sinusoid regression (|T|=1000) with 10 particles
python bmaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=1000
```

* Examples of EMAML running script
```
# 5 shot sinusoid regression (|T|=100) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=10000

# 10 shot sinusoid regression (|T|=100) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=100 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=10000

# 5 shot sinusoid regression (|T|=1000) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=5 --val_k_shot=5 --num_epochs=1000

# 10 shot sinusoid regression (|T|=1000) with 10 particles
python emaml_main.py --finite=True --train_total_num_tasks=1000 --test_total_num_tasks=100 --num_particles=10 --num_tasks=10 --few_k_shot=10 --val_k_shot=10 --num_epochs=1000
```

We tested this code on the following versions of libraries (not every ones).
* TensorFlow v.1.14
* Numpy v.1.15.4

You can get figures similar to that in the paper with plotting.py by setting configurations.

## Contact
Any feedback is welcome! Please open an issue on this repository or send email to Jaesik Yoon (jaesik817@gmail.com), Taesup Kim (taesup.kim@umontreal.ca) or Sungjin Ahn (sungjin.ahn@rutgers.edu).

