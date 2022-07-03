from typing import Any, Dict
import torch
import torch.nn as nn

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # 8, 16, 32, ... 1024
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)

    ### YOUR CODE HERE
    # TODO:
    # - define the learning rate search space [1e-5, 1] (log) -> `suggest_float`
    # - define the network architecture search space ["tiny", "small"] -> `suggest_categorical`
    # - define the activation function search space ["tanh", "relu"]
    learning_rate = ...
    net_arch = ...
    activation_fn = ...

    ### END OF YOUR CODE

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]}
        if net_arch == "tiny"
        else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        },
    }