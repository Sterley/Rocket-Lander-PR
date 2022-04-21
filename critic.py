
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch as th
from sb3m.stable_baselines3.common.utils import obs_as_tensor

import gym
import rocket_lander_gym
from stable_baselines3 import PPO


def final_show(save_figure, plot, figure_name, x_label, y_label, title, directory) -> None:
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_figure:
        directory = os.getcwd() + "/data" + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figure_name)
    if plot:
        plt.show()
    plt.close()

def plot_nd_critic(model, env, plot=True, figname="vfunction.pdf", foldername="/plots/", save_figure=True) -> None:
    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[1], state_max[1], num=definition)):
            obs = np.array([[x, y]])
            for _ in range(2, len(state_min)):
                z = random.random() - 0.5
                obs = np.append(obs, z)
            with th.no_grad():
                action = model.policy.forward(obs=obs_as_tensor(obs, model.device))
                value = model.predict_values(obs_as_tensor(obs, model.device), action)
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[1], state_max[1]], aspect="auto")
    plt.colorbar(label="critic value")
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "V Function", foldername)

def main():
    env = gym.make('RocketLander-v0')
    model = PPO.load("ppo_rkl_Top_HP2", env=env)
    plot_nd_critic(model=model, env=env)

main()
