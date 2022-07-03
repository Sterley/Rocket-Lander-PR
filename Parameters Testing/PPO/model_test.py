from stable_baselines3 import PPO
import gym
import rocket_lander_gym

def main():
    env = gym.make('RocketLander-v0')
    model = PPO.load("Best_Model_Rocket_Lander_Plus", env=env)
    l = 0
    while l < 1000:
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            print("Action Taken  ",action)
            print("Observation   ",obs)
            print("Reward Gained ",rewards)
            print("Info          ",info,end='\n\n')
            if dones:
                print("Simulation done.")
                env.close()
                break
        l +=1

main()
