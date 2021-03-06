---
name: "\U0001F916 Custom Gym Environment Issue"
about: How to report an issue when using a custom Gym environment
labels: question, custom gym env
---

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.

### 🤖 Custom Gym Environment

**Please check your environment first using**:

```python
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```

### Describe the bug

A clear and concise description of what the bug is.

### Code example

Please try to provide a minimal example to reproduce the bug.
For a custom environment, you need to give at least the observation space, action space, `reset()` and `step()` methods
(see working example below).
Error messages and stack traces are also helpful.

Please use the [markdown code blocks](https://help.github.com/en/articles/creating-and-highlighting-code-blocks)
for both code and stack traces.

```python
import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env


class CustomEnv(gym.Env):

  def __init__(self):
    super(CustomEnv, self).__init__()
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

  def reset(self):
    return self.observation_space.sample()

  def step(self, action):
    obs = self.observation_space.sample()
    reward = 1.0
    done = False
    info = {}
    return obs, reward, done, info

env = CustomEnv()
check_env(env)

model = A2C("MlpPolicy", env, verbose=1).learn(1000)
```

```bash
Traceback (most recent call last): File ...

```

### System Info
Describe the characteristic of your environment:
 * Describe how the library was installed (pip, docker, source, ...)
 * GPU models and configuration
 * Python version
 * PyTorch version
 * Gym version
 * Versions of any other relevant libraries

You can use `sb3.get_system_info()` to print relevant packages info:
```python
import stable_baselines3 as sb3
sb3.get_system_info()
```

### Additional context
Add any other context about the problem here.

### Checklist

- [ ] I have read the [documentation](https://stable-baselines3.readthedocs.io/en/master/) (**required**)
- [ ] I have checked that there is no similar [issue](https://github.com/DLR-RM/stable-baselines3/issues) in the repo (**required**)
- [ ] I have checked my env using the env checker (**required**)
- [ ] I have provided a minimal working example to reproduce the bug (**required**)
