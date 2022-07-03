from stable_baselines3 import PPO
import gym
import rocket_lander_gym

def main():
    env = gym.make('RocketLander-v0')
    """
    model = PPO(env=env, policy = 'MlpPolicy', \
    batch_size=128, \
    n_steps=1024, \
    gamma=0.999, \
    learning_rate=0.00010464624, \
    ent_coef=3.57998661e-06, \
    clip_range=0.3, \
    n_epochs=10, \
    gae_lambda=0.98, \
    max_grad_norm=2, \
    vf_coef=0.1032071476, \
    tensorboard_log="./ppo_rkl_tensorboard/"
    )
    model.learn(total_timesteps = 5000000)
    model.save("ppo_rkl_Top_HP")
    """
    model = PPO.load("ppo_rkl", env=env, policy = 'MlpPolicy', \
    batch_size=128, \
    n_steps=1024, \
    gamma=0.999, \
    learning_rate=0.00010464624, \
    ent_coef=3.57998661e-06, \
    clip_range=0.3, \
    n_epochs=10, \
    gae_lambda=0.98, \
    max_grad_norm=2, \
    vf_coef=0.1032071476, \
    tensorboard_log="./ppo_rkl_tensorboard/"
    )
    model.learn(total_timesteps = 25000000)
    model.save("ppo_rkl_Top_HP4")


main()
