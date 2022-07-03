from stable_baselines3 import PPO
import gym
import rocket_lander_gym

def main():

    env = gym.make('RocketLander-v0')
    model = PPO.load("ppo_rkl", env=env)

    obs = env.reset()
    print(obs)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    

main()
