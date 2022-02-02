from rocket_lander_gym import envs
from stable_baselines3 import PPO

if __name__ == '__main__':
    env = env = envs.RocketLander()

    model = PPO.load("ppo_rkl", env=env)

    l = 0

    while l < 100:

        obs = env.reset()
        while True:

            i = 0
            while i < 5000:
                i += 1

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
