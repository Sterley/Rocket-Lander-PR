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
import tensorflow as tf


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
            print("obs1", obsvT)
            tmp = []
            tmp.append(obsvT) 
            obsv = np.asarray(tmp)
            print("obs2", obsv)

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


def plot_policy_X_Y(policy, env, plot=True, save_figure=True) -> None:

    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    H = 385
    W = 278
    portrait = np.zeros((H, W))
    obs = env.reset()
    actions, _states = policy.predict(obs)
    obsvT, rewards, dones, info = env.step(actions)
    tmp = []
    tmp.append(obsvT)
    obsv = np.asarray(tmp)
    for y in range(H):
        for x in range(W):      
            env.lander.position.y = y 
            env.lander.position.x = x
            actions, _states = policy.predict(obsv[0])
            obsvT, rewards, dones, info = env.step(actions)
            tmp = []
            tmp.append(obsvT)
            obsv = np.asarray(tmp)
            portrait[H - (1 + int(y)), x] = actions[1]

    plt.figure(figsize=(int(H/30), int(W/30)))
    plt.imshow(portrait, cmap="inferno", extent=[0, W, 0, H], aspect="auto")
    plt.colorbar(label="throttle")

    final_show(save_figure, plot, "throttle.pdf", "Position X", "Position Y", "throttle/x/y", "/plots/")



def plot_nd_policy2(policy, env, deterministic, plot=True, figname="nd_actor.pdf", save_figure=True) -> None:

    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    for index_x, x in enumerate(np.linspace(state_min[4], state_max[4], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[3], state_max[3], num=definition)):
            obs = np.array([[]])
            for _ in range(0, 1):
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = np.append(obs, y)
            for _ in range(0, 0):
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = np.append(obs, x)
            for _ in range(3, len(state_min)):
                z = random.random() - 0.5
                obs = np.append(obs, z)
            action, _ = policy.predict(obs, deterministic=deterministic)
            #portrait[definition - (1 + index_y), index_x] = action[1]
            portrait[definition - (1 + index_y), index_x] = action[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[1], state_max[1]], aspect="auto")
    plt.colorbar(label="Cardan Moteur")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["Angle", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", "/plots/")






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





def state_reward(model, env, deterministic, plot=True, figname="nd_actor.pdf", save_figure=True) -> None:

    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    listeRewards = []
    
    for index_x, x in enumerate(np.linspace(state_min[4], state_max[4], num=definition)):
        obs = env.reset()
        obs[2] = x
        print(index_x)

        mean_reward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            mean_reward += rewards
            if dones:
                env.close()
                break
        listeRewards.append(mean_reward)



    plt.plot([i for i in range(len(listeRewards))], listeRewards, color='b')
    plt.xlabel("Angle (sur 100 pas : -1 à 1)")
    plt.ylabel("Total Rewards")

    if save_figure:
        directory = os.getcwd() + "/dataVizu/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + "plot_state_rewards")
    plt.show()
    plt.close()
    
