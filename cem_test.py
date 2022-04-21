import os
import random
from typing import TextIO, Tuple

import numpy as np
import matplotlib.pyplot as plt


import gym

import torch
from chrono import Chrono # To measure time in human readable format, use stop() to display time since chrono creation

from visu.visu_critics import plot_2d_critic # Function to plot critics
from visu.visu_policies import plot_2d_policy # Function to plot policies

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from visu.visu_trajectories import plot_trajectory
from stable_baselines3 import CEM
import rocket_lander_gym


log_dir = "data/save/"
os.makedirs(log_dir, exist_ok=True)

env_name = 'RocketLander-v0'
env = gym.make(env_name)
env_vec = make_vec_env(env_name, n_envs=10, seed=0)

file_name = env_name
log_file_name = log_dir + file_name
print (log_file_name)

eval_callback = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

model = CEM.load("data/save/best/Best_Model_Rocket_Lander_Plus", env=env)

"""
model = CEM(
            "MlpPolicy",
            env,
            seed=3,
            verbose=1,
            noise_multiplier=0.999,
            n_eval_episodes=4,
            sigma=0.2,
            pop_size=20,
            policy_kwargs=dict(net_arch=[32]),
            tensorboard_log=log_file_name,
        )
"""

model.learn(
            total_timesteps=5e6,
            callback=eval_callback,
            log_interval=20,
        )

model.save("data/save/best/cem_rkl2")
