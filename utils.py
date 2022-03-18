import os
from turtle import color
from libcst import Break
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
from sb3m.stable_baselines3 import DDPG, REINFORCE
import random
import torch as th
from sb3m.stable_baselines3.common.utils import obs_as_tensor


def get_data_list_2D(x_lab, y_lab, filename):
    data_csv = pd.read_csv(filename, delimiter=",")
    x = data_csv.loc[:,x_lab]
    y = data_csv.loc[:,y_lab]
    x = list(x)
    y = list(y)
    return x, y


def show_data(liste_x, liste_y, x_label, y_label, title, save_figure, figure_name):

    x = np.array(liste_x)
    y = np.array(liste_y)

    tmp = []
    for i in range(len(liste_x)):
        tmp.append(i)
    #plt.plot(tmp, x, label = x_label)
    #plt.plot(tmp, y, label = y_label)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)

    if save_figure:
        directory = os.getcwd() + "/dataVizu/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figure_name)
    plt.show()
    plt.close()


def save_data(title, data, filename):
    print("Saving Data...")
    fichier = open(filename,'w')
    obj = csv.writer(fichier)
    obj.writerow(title)
    for d in data:
        obj.writerow(d)
    fichier.close()
    print("Data Saved...")

def to_write(x, y, angle, gimbal, throttle, force_dir, power, rewards):
    to_write = []
    to_write.append(x)
    to_write.append(y)
    to_write.append(angle)
    to_write.append(gimbal)
    to_write.append(throttle)
    to_write.append(force_dir)
    to_write.append(power)
    to_write.append(rewards)
    return to_write


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



def plot_critic(model, env, plot=True, figname="vfunction.pdf", foldername="/plots/", save_figure=True) -> None:

    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((400, 100))

    x_rewards = []

    for x in range(100):
        obsv = env.reset()
        tmp = []
        tmp.append(obsv)
        obsv = np.asarray(tmp)
        nb_rewards = 0
        sum_rewards = 0
        while True:
            actions, _states = model.predict(obsv[0])
            obsvT, rewards, dones, info = env.step(actions)

            nb_rewards += 1
            sum_rewards += rewards

            tmp = []
            tmp.append(obsvT) 
            obsv = np.asarray(tmp)

            value = model.policy.predict_values(obs_as_tensor(obsv, model.device))
            value = value[0][0]
            portrait[400 - (1 + int(env.lander.position.y)), x] = value.item()
            if dones:
                break
        mean_rewars = sum_rewards/nb_rewards
        tmp = []
        tmp.append(x)
        tmp.append(mean_rewars)
        x_rewards.append(tmp)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[0, 100, 0, 400], aspect="auto")
    plt.colorbar(label="Value")
    min_r = x_rewards[0][1]
    max_r = x_rewards[0][1]
    for xr in x_rewards:
        r = xr[1]
        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r
        if r < 0:
            r = r*-1
        r = r*200
        plt.scatter(xr[0],r*15,color='b')
    final_show(save_figure, plot, figname, "Nombre de Env.reset()", "Position Y", "Rewards (Min: "+str(min_r)+", Max: "+str(max_r)+")", foldername)



def plot_policy(policy, env, plot=True, save_figure=True) -> None:

    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((400, 100))
    
    x_rewards = []

    for x in range(100):
        obsv = env.reset()
        tmp = []
        tmp.append(obsv)
        obsv = np.asarray(tmp)
        nb_rewards = 0
        sum_rewards = 0
        while True:
            actions, _states = policy.predict(obsv[0])
            obsvT, rewards, dones, info = env.step(actions)

            nb_rewards += 1
            sum_rewards += rewards

            tmp = []
            tmp.append(obsvT)
            obsv = np.asarray(tmp)
            portrait[400 - (1 + int(env.lander.position.y)), x] = actions[1]
            # gimbal action[0] 
            # throttle action[1]
            # force_dir action[2]
            if dones:
                break
        mean_rewars = sum_rewards/nb_rewards
        tmp = []
        tmp.append(x)
        tmp.append(mean_rewars)
        x_rewards.append(tmp)

    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[0, 100, 0, 400], aspect="auto")
    plt.colorbar(label="throttle")

    min_r = x_rewards[0][1]
    max_r = x_rewards[0][1]
    for xr in x_rewards:
        r = xr[1]
        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r
        if r < 0:
            r = r*-1
        r = r*200
        plt.scatter(xr[0],r*15,color='b')

    final_show(save_figure, plot, "throttle.pdf", "Nombre de Env.reset()", "Position Y", "Rewards (Min: "+str(min_r)+", Max: "+str(max_r)+")", "/plots/")


def episode_to_traj(rollout_data):
    """
    Transform the states of a rollout_data into a set of (x,y) pairs
    :param rollout_data:
    :return: the (x,y) pairs
    """
    x = []
    y = []
    # print("rd", rollout_data)
    obs = rollout_data.observations
    # print("obs", obs)
    # TODO : treat the case where the variables to plot are not the first two
    for o in obs:
        x.append(o[0].numpy())
        y.append(o[1].numpy())
    return x, y


def plot_trajectory(rollout_data, env, fig_index, save_figure=True, plot=True) -> None:
    """
    Plot the set of trajectories stored into a batch
    :param rollout_data: the source batch
    :param env: the environment where the batch was built
    :param fig_index: a number, to save several similar plots
    :param save_figure: where the plot should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] < 2:
        raise (ValueError("Observation space of dimension {}, should be at least 2".format(env.observation_space.shape[0])))

    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    x, y = episode_to_traj(rollout_data)
    plt.scatter(x, y, c=range(1, len(rollout_data.observations) + 1), s=3)
    figname = "trajectory_" + str(fig_index) + ".pdf"
    final_show(save_figure, plot, figname, x_label, y_label, "Trajectory", "/plots/")