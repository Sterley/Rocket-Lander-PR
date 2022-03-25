from gc import callbacks
import rlcompleter
import gym
import rocket_lander_gym
from stable_baselines3 import PPO
import utils
from stable_baselines3.common.callbacks import EvalCallback


def main():
    env = gym.make('RocketLander-v0')
    model = PPO.load("ppo_rkl_Top_HP2", env=env)
    #model2 = PPO("MlpPolicy", env=env)
    utils.plot_policy_X_Y(model.policy, env=env)
    #utils.plot_critic(model=model2, env=env)

main()


