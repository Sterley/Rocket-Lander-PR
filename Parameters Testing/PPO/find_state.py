from stable_baselines3 import PPO
import gym
import rocket_lander_gym
import matplotlib.pyplot as plt

def main():

    env = gym.make('RocketLander-v0')
    model = PPO.load("Best_Model_Rocket_Lander_Plus", env=env)

    liste_state = []
    liste_rewards = []

    state_angle = -1
    while state_angle < 1:
        state_angle += 0.01
        total_rewards = 0

        l = 0
        while l < 20:

            env.lander.angle = state_angle
            obs = env.reset()

            while True:

                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)

                total_rewards += rewards

                if dones:
                    env.close()
                    break

            l +=1

        print(state_angle, total_rewards)
        liste_state.append(state_angle)
        liste_rewards.append(total_rewards/20)

    plt.plot(liste_state,liste_rewards)  
    plt.ylabel('Rewards')
    plt.xlabel("Angle")
    plt.show()




main()
