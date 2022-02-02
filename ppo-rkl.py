from rocket_lander_gym import envs
from stable_baselines3 import PPO

if __name__ == '__main__':
    env = env = envs.RocketLander()

    model = PPO.load("ppo_rkl", env=env, tensorboard_log="./ppo_rkl_tensorboard/")

    l = 0

    while l < 100:

        model.learn(total_timesteps=int(2e5))

        model.save("ppo_rkl")

        obs = env.reset()
        while True:

            i = 0
            while i < 10000:
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
