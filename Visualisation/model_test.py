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
    #utils.plot_nd_policy2(model.policy, env=env, deterministic=True)
    #utils.plot_critic(model=model, env=env)
    #utils.state_reward(model, env=env)
    #utils.plot_traj_rew(model, env=env, deterministic=True)
    #utils.plot_nd_critic(model=model, env=env)
    
    
    

main()


